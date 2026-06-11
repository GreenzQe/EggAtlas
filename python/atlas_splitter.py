#!/usr/bin/env python3
"""
Texture Atlas Splitter
Extracts individual sprites from a PNG texture atlas.

Install:
  pip install Pillow numpy

Input modes (pick one):
  --meta <file>
      Load sprite rectangles from a metadata file.
      Supported formats:
        .json   TexturePacker JSON Hash or JSON Array
        .xml    Starling / Cocos2d XML
        .atlas  LibGDX text atlas
      Handles rotated and trimmed sprites automatically.

  --auto
      Detect sprites by finding connected non-transparent regions.
      Each disconnected blob of pixels becomes its own sprite.
      Options:
        --threshold 0-255   Alpha cutoff (default 10)
        --min-size PX       Ignore blobs smaller than this (default 4)

  --auto --cell N
      Same as --auto but scans in N×N steps and merges adjacent cells
      when pixels touch the shared edge. Use this when a single sprite
      has multiple blobs separated by transparency (e.g. particle effects).
      N must be a multiple of 8.

  --grid N
      Split the atlas into uniform N×N square tiles.
      N must be a multiple of 8.

Output options:
  -o / --output <dir>   Output folder (default: sprites/)
  --snap                Round each sprite up to the nearest square
                        power-of-2 canvas and center the content
                        (e.g. 56×50 → 64×64, 200×210 → 256×256)
  --keep-empty          Also save fully transparent sprites

Examples:
  python atlas_splitter.py atlas.png --meta atlas.json -o sprites/
  python atlas_splitter.py atlas.png --auto -o sprites/
  python atlas_splitter.py atlas.png --auto --cell 8 --snap -o sprites/
  python atlas_splitter.py atlas.png --grid 32 -o tiles/
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Metadata parsers
# ---------------------------------------------------------------------------

def parse_json(path):
    """TexturePacker JSON Hash or Array format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", {})
    sprites = {}

    if isinstance(frames, dict):
        items = frames.items()
    elif isinstance(frames, list):
        items = ((entry["filename"], entry) for entry in frames)
    else:
        sys.exit(f"Unrecognised JSON structure in {path}")

    for name, info in items:
        fr = info["frame"]
        sprites[name] = {
            "x": fr["x"], "y": fr["y"], "w": fr["w"], "h": fr["h"],
            "rotated": info.get("rotated", False),
            "trimmed": info.get("trimmed", False),
            "sourceSize": info.get("sourceSize"),
            "spriteSourceSize": info.get("spriteSourceSize"),
        }

    return sprites


def parse_xml(path):
    """TexturePacker XML / Starling / Cocos2d format."""
    root = ET.parse(path).getroot()
    sprites = {}

    for child in root:
        tag = child.tag.lower()
        if tag not in ("subtexture", "sprite"):
            continue
        name = child.get("name") or child.get("n", "unknown")
        x = int(child.get("x", 0))
        y = int(child.get("y", 0))
        w = int(child.get("width") or child.get("w", 0))
        h = int(child.get("height") or child.get("h", 0))
        rotated = child.get("r") in ("y", "true") or child.get("rotated", "false").lower() == "true"
        sprites[name] = {
            "x": x, "y": y, "w": w, "h": h,
            "rotated": rotated, "trimmed": False,
            "sourceSize": None, "spriteSourceSize": None,
        }

    return sprites


def parse_libgdx(path):
    """LibGDX / libGDX .atlas text format."""
    sprites = {}
    current = None

    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip() for l in f]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Blank lines separate sections; skip page header lines (image filename)
        if not line or line.endswith(".png") or line.endswith(".jpg"):
            i += 1
            continue

        # Key: value pairs belong to the current sprite
        if ": " in line:
            key, _, value = line.partition(": ")
            key = key.strip()
            value = value.strip()

            if current is not None:
                if key == "xy":
                    x, y = map(int, value.split(","))
                    sprites[current]["x"] = x
                    sprites[current]["y"] = y
                elif key == "size":
                    w, h = map(int, value.split(","))
                    sprites[current]["w"] = w
                    sprites[current]["h"] = h
                elif key == "rotate":
                    sprites[current]["rotated"] = value.lower() == "true"
        else:
            # New sprite name
            current = line.strip()
            sprites[current] = {
                "x": 0, "y": 0, "w": 0, "h": 0,
                "rotated": False, "trimmed": False,
                "sourceSize": None, "spriteSourceSize": None,
            }

        i += 1

    return sprites


def load_metadata(meta_path):
    p = Path(meta_path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return parse_json(p)
    elif suffix in (".xml", ".plist"):
        return parse_xml(p)
    elif suffix == ".atlas":
        return parse_libgdx(p)
    else:
        # Try JSON first, then XML
        try:
            return parse_json(p)
        except (json.JSONDecodeError, KeyError):
            pass
        try:
            return parse_xml(p)
        except ET.ParseError:
            pass
        sys.exit(f"Could not parse metadata file: {meta_path}")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def auto_detect_numpy(img, alpha_threshold=10, min_size=4):
    """Connected-component labelling on the alpha channel."""
    try:
        from scipy.ndimage import label as nd_label
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img)[:, :, 3]
    mask = alpha > alpha_threshold

    if HAS_SCIPY:
        labeled, n = nd_label(mask)
    else:
        # BFS-based connected components using numpy
        labeled, n = _numpy_label(mask)

    sprites = {}
    for label_id in range(1, n + 1):
        ys, xs = np.where(labeled == label_id)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w < min_size or h < min_size:
            continue
        sprites[f"sprite_{len(sprites):04d}"] = {
            "x": x0, "y": y0, "w": w, "h": h,
            "rotated": False, "trimmed": False,
            "sourceSize": None, "spriteSourceSize": None,
        }

    return sprites


def _numpy_label(mask):
    """Pure-numpy BFS connected components (8-connected)."""
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    seed_ys, seed_xs = np.where(mask & (labeled == 0))
    seed_pairs = list(zip(seed_ys.tolist(), seed_xs.tolist()))

    for sy, sx in seed_pairs:
        if labeled[sy, sx] != 0 or not mask[sy, sx]:
            continue
        current_label += 1
        queue = [(sy, sx)]
        labeled[sy, sx] = current_label
        head = 0
        while head < len(queue):
            cy, cx = queue[head]
            head += 1
            for dy, dx in offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and labeled[ny, nx] == 0 and mask[ny, nx]:
                    labeled[ny, nx] = current_label
                    queue.append((ny, nx))

    return labeled, current_label


def auto_detect_pil(img, alpha_threshold=10, min_size=4):
    """
    PIL-only fallback: uses row/column projections to find rectangular sprite
    regions. Works well when sprites are separated by fully transparent gaps.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    width, height = img.size
    pixels = img.load()

    # Build row and column opacity flags
    row_has_pixel = [False] * height
    col_has_pixel = [False] * width

    for y in range(height):
        for x in range(width):
            if pixels[x, y][3] > alpha_threshold:
                row_has_pixel[y] = True
                col_has_pixel[x] = True

    def find_spans(flags, min_gap=1):
        spans = []
        in_span = False
        start = 0
        for i, v in enumerate(flags):
            if v and not in_span:
                in_span = True
                start = i
            elif not v and in_span:
                in_span = False
                if i - start >= min_size:
                    spans.append((start, i))
        if in_span and len(flags) - start >= min_size:
            spans.append((start, len(flags)))
        return spans

    row_spans = find_spans(row_has_pixel)
    col_spans = find_spans(col_has_pixel)

    # Intersect row bands × col bands to find bounding boxes
    sprites = {}
    for y0, y1 in row_spans:
        for x0, x1 in col_spans:
            # Check if this cell actually contains pixels
            has_content = any(
                pixels[x, y][3] > alpha_threshold
                for y in range(y0, y1)
                for x in range(x0, x1)
            )
            if has_content:
                name = f"sprite_{len(sprites):04d}"
                sprites[name] = {
                    "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                    "rotated": False, "trimmed": False,
                    "sourceSize": None, "spriteSourceSize": None,
                }

    return sprites


def auto_detect_edge_cells(img, cell_size, alpha_threshold=10):
    """
    Scan the atlas in cell_size×cell_size steps left-to-right, top-to-bottom.
    Two adjacent cells are part of the same sprite when either cell has pixels
    touching their shared edge — meaning the content bleeds across the boundary.
    Cells with content but no edge-touching pixels are self-contained sprites.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    width, height = img.size
    alpha = (np.array(img)[:, :, 3] > alpha_threshold)

    cols = (width  + cell_size - 1) // cell_size
    rows = (height + cell_size - 1) // cell_size

    # Per-cell flags
    has_content = {}
    touches_right  = {}
    touches_bottom = {}

    for row in range(rows):
        for col in range(cols):
            y0 = row * cell_size
            y1 = min(y0 + cell_size, height)
            x0 = col * cell_size
            x1 = min(x0 + cell_size, width)
            cell = alpha[y0:y1, x0:x1]

            has_content[(row, col)] = bool(cell.any())
            touches_right[(row, col)]  = bool(cell[:, -1].any())
            touches_bottom[(row, col)] = bool(cell[-1, :].any())

    # Union-Find
    parent = {(r, c): (r, c) for r in range(rows) for c in range(cols)}

    def find(pos):
        while parent[pos] != pos:
            parent[pos] = parent[parent[pos]]
            pos = parent[pos]
        return pos

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in range(rows):
        for col in range(cols):
            if not has_content[(row, col)]:
                continue
            # Merge with right neighbour only when BOTH cells touch the shared edge
            if col + 1 < cols and has_content[(row, col + 1)]:
                ny0 = row * cell_size
                ny1 = min(ny0 + cell_size, height)
                nx0 = (col + 1) * cell_size
                right_neighbour_touches_left = bool(alpha[ny0:ny1, nx0:nx0 + 1].any())
                if touches_right[(row, col)] and right_neighbour_touches_left:
                    union((row, col), (row, col + 1))
            # Merge with bottom neighbour only when BOTH cells touch the shared edge
            if row + 1 < rows and has_content[(row + 1, col)]:
                bx0 = col * cell_size
                bx1 = min(bx0 + cell_size, width)
                by0 = (row + 1) * cell_size
                bottom_neighbour_touches_top = bool(alpha[by0:by0 + 1, bx0:bx1].any())
                if touches_bottom[(row, col)] and bottom_neighbour_touches_top:
                    union((row, col), (row + 1, col))

    # Group cells by component root
    groups = {}
    for r in range(rows):
        for c in range(cols):
            if not has_content[(r, c)]:
                continue
            groups.setdefault(find((r, c)), []).append((r, c))

    sprites = {}
    for cells in groups.values():
        px_x0 = min(c * cell_size for _, c in cells)
        px_y0 = min(r * cell_size for r, _ in cells)
        px_x1 = min(max((c + 1) * cell_size for _, c in cells), width)
        px_y1 = min(max((r + 1) * cell_size for r, _ in cells), height)
        sprites[f"sprite_{len(sprites):04d}"] = {
            "x": px_x0, "y": px_y0,
            "w": px_x1 - px_x0, "h": px_y1 - px_y0,
            "rotated": False, "trimmed": False,
            "sourceSize": None, "spriteSourceSize": None,
        }

    return sprites


def auto_detect(img, alpha_threshold=10, min_size=4, cell_size=None):
    if cell_size is not None:
        if not HAS_NUMPY:
            sys.exit("numpy is required for --cell mode: pip install numpy")
        return auto_detect_edge_cells(img, cell_size, alpha_threshold)
    if HAS_NUMPY:
        return auto_detect_numpy(img, alpha_threshold, min_size)
    else:
        print("numpy not found — using PIL projection method (may miss overlapping sprites)")
        return auto_detect_pil(img, alpha_threshold, min_size)


# ---------------------------------------------------------------------------
# Grid mode
# ---------------------------------------------------------------------------

def grid_sprites(img_width, img_height, cell_w, cell_h):
    sprites = {}
    cols = img_width // cell_w
    rows = img_height // cell_h
    for row in range(rows):
        for col in range(cols):
            name = f"tile_r{row:03d}_c{col:03d}"
            sprites[name] = {
                "x": col * cell_w, "y": row * cell_h,
                "w": cell_w, "h": cell_h,
                "rotated": False, "trimmed": False,
                "sourceSize": None, "spriteSourceSize": None,
            }
    return sprites


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def safe_filename(name):
    """Strip path separators and sanitise for use as a filename."""
    p = Path(name)
    stem = p.stem if p.suffix else name
    bad = r'\/:*?"<>|'
    return "".join("_" if c in bad else c for c in stem)


def next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def snap_to_po2(sprite):
    """Return a new square power-of-2 canvas with the sprite centered."""
    w, h = sprite.size
    size = next_power_of_two(max(w, h))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2
    canvas.paste(sprite, (offset_x, offset_y))
    return canvas


def extract_sprite(atlas, info):
    """Crop a sprite from the atlas, handling rotation and transparent restore."""
    x, y, w, h = info["x"], info["y"], info["w"], info["h"]
    rotated = info.get("rotated", False)

    # When rotated, TexturePacker stores (w, h) as the rotated dimensions
    if rotated:
        region = atlas.crop((x, y, x + h, y + w))
        region = region.transpose(Image.ROTATE_90)
    else:
        region = atlas.crop((x, y, x + w, y + h))

    # Restore original canvas size for trimmed sprites
    src_size = info.get("sourceSize")
    sss = info.get("spriteSourceSize")
    if info.get("trimmed") and src_size and sss:
        canvas = Image.new("RGBA", (src_size["w"], src_size["h"]), (0, 0, 0, 0))
        canvas.paste(region, (sss["x"], sss["y"]))
        return canvas

    return region


def extract_all(atlas_path, sprites, output_dir, skip_empty=True, snap=False):
    atlas = Image.open(atlas_path).convert("RGBA")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    for name, info in sprites.items():
        sprite = extract_sprite(atlas, info)

        if skip_empty:
            bbox = sprite.getbbox()
            if bbox is None:
                skipped += 1
                continue

        if snap:
            sprite = snap_to_po2(sprite)

        fname = safe_filename(name) + ".png"
        out_path = output_dir / fname

        # Avoid silent overwrites when names collide after sanitisation
        if out_path.exists():
            stem = out_path.stem
            n = 1
            while out_path.exists():
                out_path = output_dir / f"{stem}_{n}.png"
                n += 1

        sprite.save(out_path, "PNG")
        saved += 1
        size_str = f"{sprite.width}x{sprite.height}"
        print(f"  saved {out_path.name}  ({size_str})")

    print(f"\nDone — {saved} sprites saved, {skipped} empty sprites skipped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split a PNG texture atlas into individual sprite PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("atlas", help="Input PNG texture atlas")
    parser.add_argument("-o", "--output", default="sprites", help="Output directory (default: sprites/)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--meta", metavar="FILE",
                      help="Metadata file: .json (TexturePacker), .xml (Starling/Cocos2d), .atlas (LibGDX)")
    mode.add_argument("--auto", action="store_true",
                      help="Auto-detect sprites via alpha transparency")
    mode.add_argument("--grid", metavar="N",
                      help="Split into square grid cells; N must be a multiple of 8 (e.g. 32)")

    parser.add_argument("--threshold", type=int, default=10, metavar="0-255",
                        help="Alpha threshold for auto-detection (default: 10)")
    parser.add_argument("--min-size", type=int, default=4, metavar="PX",
                        help="Minimum sprite dimension in pixels for auto-detection (default: 4)")
    parser.add_argument("--cell", type=int, default=None, metavar="N",
                        help="Coarse-grid cell size for --auto (must be multiple of 8). "
                             "Blobs within the same NxN cell are merged into one sprite. "
                             "Use when a single sprite has multiple disconnected parts.")
    parser.add_argument("--keep-empty", action="store_true",
                        help="Save sprites that are fully transparent")
    parser.add_argument("--snap", action="store_true",
                        help="Snap each sprite to a square power-of-2 canvas (e.g. 56x50 → 64x64), centered")

    args = parser.parse_args()

    atlas_path = Path(args.atlas)
    if not atlas_path.exists():
        sys.exit(f"Atlas file not found: {atlas_path}")

    print(f"Atlas: {atlas_path}")

    if args.meta:
        print(f"Mode:  metadata ({args.meta})")
        sprites = load_metadata(args.meta)
    elif args.auto:
        cell = args.cell
        if cell is not None:
            if cell <= 0 or cell % 8 != 0:
                sys.exit(f"--cell must be a positive multiple of 8 (got {cell})")
            print(f"Mode:  auto-detect (coarse grid {cell}x{cell})")
        else:
            print("Mode:  auto-detect (pixel-level)")
        img = Image.open(atlas_path)
        sprites = auto_detect(img, args.threshold, args.min_size, cell_size=cell)
    else:
        try:
            cell_size = int(args.grid)
        except ValueError:
            sys.exit("--grid must be a single integer, e.g. 32")
        if cell_size <= 0 or cell_size % 8 != 0:
            sys.exit(f"--grid value must be a positive multiple of 8 (got {cell_size})")
        cell_w = cell_h = cell_size
        print(f"Mode:  grid ({cell_w}x{cell_h})")
        with Image.open(atlas_path) as img:
            sprites = grid_sprites(img.width, img.height, cell_w, cell_h)

    if not sprites:
        sys.exit("No sprites found. Check your metadata file or try adjusting --threshold.")

    print(f"Found: {len(sprites)} sprite(s)")
    print(f"Output: {Path(args.output).resolve()}\n")

    extract_all(atlas_path, sprites, args.output, skip_empty=not args.keep_empty, snap=args.snap)


if __name__ == "__main__":
    main()
