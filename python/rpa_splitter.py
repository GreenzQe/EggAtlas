#!/usr/bin/env python3
"""
RPA Atlas Splitter
Cuts sprites from a PNG texture atlas using a custom .rpa coordinate file.

File format (binary, little-endian):
  - 4 bytes: uint32 entry count
  - N entries of 524 bytes each:
      bytes   0–N:  null-terminated ASCII sprite name
      bytes 508–511: float32 = 0.0  (unused)
      bytes 512–515: float32 = x_left UV
      bytes 516–519: float32 = y_top UV
      bytes 520–523: float32 = size UV  (sprites are square: width == height)

Usage:
  python rpa_splitter.py atlas.png coords.rpa -o output/
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")


def parse_rpa(path):
    data = Path(path).read_bytes()
    if len(data) < 4:
        sys.exit(f"RPA file too small: {path}")

    count = struct.unpack_from("<I", data, 0)[0]
    if count == 0:
        sys.exit("RPA file contains 0 entries.")

    remaining = len(data) - 4
    if remaining % count != 0:
        sys.exit(
            f"File size {len(data)} is not consistent with {count} entries "
            f"({remaining} bytes remaining, not divisible by {count})."
        )
    entry_size = remaining // count

    if entry_size < 524:
        sys.exit(f"Entry size {entry_size} is too small (expected ≥ 524 bytes).")

    sprites = []
    for i in range(count):
        offset = 4 + i * entry_size
        entry = data[offset : offset + entry_size]

        try:
            null_pos = entry.index(b"\x00")
        except ValueError:
            print(f"Warning: entry {i} has no null terminator — skipping.", file=sys.stderr)
            continue

        name = entry[:null_pos].decode("ascii", errors="replace").strip()
        if not name:
            continue

        # Coordinates at the last 16 bytes (4 × float32 LE)
        # layout: (unused=0, x_left, y_top, size)  — sprites are square
        _, x_left_uv, y_top_uv, size_uv = struct.unpack_from("<4f", entry, entry_size - 16)
        sprites.append(
            {
                "name": name,
                "y_top_uv": y_top_uv,
                "x_left_uv": x_left_uv,
                "size_uv": size_uv,
            }
        )

    return sprites


def compute_pixel_rects(sprites, atlas_w, atlas_h):
    result = []
    for s in sprites:
        x = round(s["x_left_uv"] * atlas_w)
        y = round(s["y_top_uv"] * atlas_h)
        size = round(s["size_uv"] * atlas_h)
        result.append({"name": s["name"], "x": x, "y": y, "w": size, "h": size})
    return result


def safe_filename(name):
    bad = r'\/:*?"<>|'
    return "".join("_" if c in bad else c for c in name)


def extract_all(atlas_path, rects, output_dir, skip_empty=True):
    atlas = Image.open(atlas_path).convert("RGBA")
    atlas_w, atlas_h = atlas.size

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0

    for r in rects:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]

        if w <= 0 or h <= 0:
            print(f"  skip {r['name']} — zero/negative size ({w}×{h})")
            skipped += 1
            continue

        # Clamp to atlas bounds
        x2 = min(x + w, atlas_w)
        y2 = min(y + h, atlas_h)
        sprite = atlas.crop((x, y, x2, y2))

        if skip_empty and sprite.getbbox() is None:
            skipped += 1
            continue

        fname = safe_filename(r["name"]) + ".png"
        out_path = out / fname

        # Avoid silent overwrites on name collision
        if out_path.exists():
            stem = out_path.stem
            n = 1
            while out_path.exists():
                out_path = out / f"{stem}_{n}.png"
                n += 1

        sprite.save(out_path, "PNG")
        saved += 1
        print(f"  saved {out_path.name}  ({sprite.width}×{sprite.height}  at {x},{y})")

    print(f"\nDone — {saved} sprites saved, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract sprites from a PNG atlas using a .rpa coordinate file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("atlas", help="Input PNG texture atlas")
    parser.add_argument("rpa", help="RPA coordinate file (.rpa)")
    parser.add_argument("-o", "--output", default="sprites", help="Output directory (default: sprites/)")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Save sprites that are fully transparent",
    )
    args = parser.parse_args()

    atlas_path = Path(args.atlas)
    rpa_path = Path(args.rpa)

    if not atlas_path.exists():
        sys.exit(f"Atlas not found: {atlas_path}")
    if not rpa_path.exists():
        sys.exit(f"RPA file not found: {rpa_path}")

    with Image.open(atlas_path) as img:
        atlas_w, atlas_h = img.size

    print(f"Atlas : {atlas_path}  ({atlas_w}×{atlas_h})")
    print(f"RPA   : {rpa_path}")
    print(f"Output: {Path(args.output).resolve()}\n")

    sprites = parse_rpa(rpa_path)
    print(f"Parsed {len(sprites)} sprite entries.")

    rects = compute_pixel_rects(sprites, atlas_w, atlas_h)
    print(f"Computed {len(rects)} pixel rects.\n")

    extract_all(atlas_path, rects, args.output, skip_empty=not args.keep_empty)


if __name__ == "__main__":
    main()
