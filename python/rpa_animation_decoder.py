#!/usr/bin/env python3
"""
RPA Animation Decoder
Decodes animation keyframe data from .rpa animation files.

File format (binary, little-endian):
  Header (variable size):
    - 4 bytes: magic "RPA1"
    - 4 bytes: uint32 num_tracks
    - Per-track descriptor (12 bytes each, starting at offset 8):
        - 4 bytes: uint32 track_type  (1 = first/X-axis track, 3 = standard value track)
        - 4 bytes: uint32 padding (0)
        - 4 bytes: uint32 keyframe_count
    - End marker / padding (4 bytes for single-track, 20 bytes for multi-track)
    - Single-track: data immediately follows
    - Multi-track: 4 extra header bytes precede keyframe data

  Keyframe data (16 bytes each, per-track):
    - 4 bytes: float32 time (seconds)
    - 4 bytes: float32 value
    - 4 bytes: float32 tangent_in
    - 4 bytes: float32 tangent_out
    Note: last keyframe of the last track in multi-track files is 12 bytes (tout omitted)

  Rest-pose entry (per track, stored at the END of each track's keyframes):
    Multi-track files append one extra keyframe with time=0 at the end of each track's
    data. This encodes the rest/default value for the property. It is NOT an animation
    keyframe and appears after all timed keyframes.

Usage:
  python rpa_animation_decoder.py animation.rpa
  python rpa_animation_decoder.py animation.rpa --plot
  python rpa_animation_decoder.py animation.rpa --output out.json
  python rpa_animation_decoder.py --folder animations/
  python rpa_animation_decoder.py --folder animations/ --plot
"""

import argparse
import json
import struct
import sys
from pathlib import Path


def decode_rpa_animation(path: Path) -> dict:
    data = path.read_bytes()

    if len(data) < 8:
        sys.exit(f"File too small: {path}")

    magic = data[0:4].decode("ascii", errors="replace")
    if magic != "RPA1":
        sys.exit(f"Not an RPA animation file (magic={magic!r}): {path}")

    num_tracks = struct.unpack_from("<I", data, 4)[0]
    if num_tracks == 0:
        sys.exit("No tracks in file.")

    # Per-track descriptors start at offset 8, each 12 bytes:
    # (track_type uint32, padding uint32, keyframe_count uint32)
    tracks_meta = []
    for i in range(num_tracks):
        base = 8 + i * 12
        track_type, _, kf_count = struct.unpack_from("<III", data, base)
        tracks_meta.append({"type": track_type, "keyframe_count": kf_count})

    total_kf = sum(t["keyframe_count"] for t in tracks_meta)

    # Multi-track files have 4 extra header bytes before the keyframe data,
    # and their last track's last KF is stored as 12 bytes (tout field omitted).
    # Single-track files have no such offset shift.
    is_multi = num_tracks > 1
    formula_offset = len(data) - total_kf * 16
    data_offset = formula_offset + (4 if is_multi else 0)

    if data_offset < 0:
        sys.exit(f"File size {len(data)} too small for declared keyframe data.")

    tracks = []
    offset = data_offset
    for track_idx, meta in enumerate(tracks_meta):
        kf_count = meta["keyframe_count"]
        is_last_track = is_multi and (track_idx == num_tracks - 1)
        keyframes = []

        for kf_idx in range(kf_count):
            # The very last KF of the last track in multi-track files is 12 bytes.
            is_partial = is_last_track and (kf_idx == kf_count - 1)

            if offset + 12 > len(data):
                break  # truncated file — stop gracefully

            time, value, tin = struct.unpack_from("<fff", data, offset)
            if is_partial or offset + 16 > len(data):
                tout = 0.0
                offset += 12
            else:
                (tout,) = struct.unpack_from("<f", data, offset + 12)
                offset += 16

            keyframes.append({
                "time": round(time, 7),
                "value": round(value, 7),
                "tangent_in": round(tin, 7),
                "tangent_out": round(tout, 7),
            })

        tracks.append({
            "type": meta["type"],
            "keyframe_count": kf_count,
            "keyframes": keyframes,
        })

    return {
        "file": str(path),
        "magic": magic,
        "num_tracks": num_tracks,
        "tracks": tracks,
    }


def _animation_keyframes(track: dict) -> list:
    """Return only timed animation KFs, filtering out rest-pose entries (time=0 at end)."""
    kfs = track["keyframes"]
    # A rest-pose KF has time=0 and appears at the END of the list after normal KFs.
    # Normal animation may also start with time=0 (single-track), so only filter if
    # there are preceding non-zero-time keyframes.
    if len(kfs) > 1 and kfs[-1]["time"] == 0.0 and kfs[-2]["time"] > 0.0:
        return kfs[:-1]
    return kfs


def plot_tracks(result: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for plotting: pip install matplotlib")

    name = Path(result["file"]).stem
    tracks = result["tracks"]
    n = len(tracks)

    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(name, fontsize=14)

    for i, track in enumerate(tracks):
        ax = axes[i][0]
        kfs = _animation_keyframes(track)

        times = [kf["time"] for kf in kfs]
        values = [kf["value"] for kf in kfs]

        ax.plot(times, values, linewidth=1.5)
        ax.set_title(f"Track {i}  (type={track['type']}, {track['keyframe_count']} keyframes)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(result["file"]).with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved: {out_path}")
    plt.show()


def print_summary(result: dict) -> None:
    print(f"File    : {result['file']}")
    print(f"Magic   : {result['magic']}")
    print(f"Tracks  : {result['num_tracks']}")
    for i, track in enumerate(result["tracks"]):
        kfs = _animation_keyframes(track)
        times = [kf["time"] for kf in kfs]
        values = [kf["value"] for kf in kfs]
        duration = max(times) if times else 0.0
        val_min = min(values) if values else 0.0
        val_max = max(values) if values else 0.0
        has_rest = len(track["keyframes"]) > len(kfs)
        rest_note = f"  rest_pose={track['keyframes'][-1]['value']:.4f}" if has_rest else ""
        print(
            f"  Track {i}: type={track['type']}, {track['keyframe_count']} kf, "
            f"duration={duration:.4f}s, value=[{val_min:.4f}, {val_max:.4f}]{rest_note}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Decode RPA animation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("rpa", nargs="?", help="RPA animation file (.rpa)")
    group.add_argument("--folder", "-f", help="Decode all .rpa files in this folder, writing .json alongside each")
    parser.add_argument("--output", "-o", help="Write decoded JSON to this file (single-file mode only)")
    parser.add_argument("--plot", action="store_true", help="Plot and save animation curves as .png")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            sys.exit(f"Not a directory: {folder}")
        files = sorted(folder.glob("*.rpa"))
        if not files:
            sys.exit(f"No .rpa files found in {folder}")
        for path in files:
            result = decode_rpa_animation(path)
            print_summary(result)
            out = path.with_suffix(".json")
            out.write_text(json.dumps(result, indent=2))
            print(f"  -> {out.name}")
            if args.plot:
                plot_tracks(result)
            print()
        return

    path = Path(args.rpa)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    result = decode_rpa_animation(path)
    print_summary(result)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(result, indent=2))
        print(f"JSON written: {out}")

    if args.plot:
        plot_tracks(result)
    elif not args.output:
        print()
        for i, track in enumerate(result["tracks"]):
            kfs = track["keyframes"]
            preview = kfs[:3] + (["..."] if len(kfs) > 6 else []) + kfs[-3:]
            print(f"Track {i} preview:")
            for kf in preview:
                if isinstance(kf, str):
                    print(f"  {kf}")
                else:
                    print(
                        f"  t={kf['time']:.5f}  v={kf['value']:.6f}"
                        f"  tin={kf['tangent_in']:.6f}  tout={kf['tangent_out']:.6f}"
                    )


if __name__ == "__main__":
    main()
