"""Download pre-trained detector weights for DeepStream-VLM.

Ultralytics publishes YOLOE only as the seg checkpoint (no detect-only `.pt`);
the same ``yoloe-26{size}-seg.pt`` file feeds BOTH export scripts —
``export_yoloe.py`` strips mask coefficients for detection-only inference,
and ``export_yoloe_seg.py`` keeps them for instance segmentation. Pick the
script based on the nvinfer config you want to use at runtime.

    --model yolo26   → yolo26{size}.pt             (closed-vocab, 80 COCO classes)
    --model yoloe    → yoloe-26{size}-seg.pt       (open-vocab, used for both
                                                    detection and segmentation)

Usage:
    python3 scripts/download_model.py --model yolo26 --size m
    python3 scripts/download_model.py --model yoloe  --size m
"""

import argparse
import os
import sys
import urllib.request

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
VALID_SIZES = ["s", "m", "l"]
VALID_MODELS = ["yolo26", "yoloe"]


def _filename_for(model: str, size: str) -> str:
    if model == "yolo26":
        return f"yolo26{size}.pt"
    if model == "yoloe":
        # Ultralytics publishes YOLOE only as the seg checkpoint.
        return f"yoloe-26{size}-seg.pt"
    raise ValueError(f"Unknown model: {model}")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(
            f"\r  [{bar}] {pct:5.1f}%  ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)",
            end="",
            flush=True,
        )


def download_model(model: str, size: str, output_dir: str) -> str:
    filename = _filename_for(model, size)
    url = f"{BASE_URL}/{filename}"
    output_path = os.path.join(output_dir, filename)

    if os.path.isfile(output_path):
        print(f"Already exists: {output_path}")
        return output_path

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {filename} ...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path, _progress_hook)
        print(f"\nDone: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"\nFailed to download {filename}: {e}", file=sys.stderr)
        sys.exit(1)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download detector weights")
    parser.add_argument(
        "--model",
        type=str,
        choices=VALID_MODELS,
        default="yolo26",
        help="Detector family: yolo26 (default) | yoloe (seg checkpoint; "
        "used for both detection and segmentation exports)",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=VALID_SIZES,
        default="m",
        help="Model size: s | m (default) | l",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/models",
        help="Output directory (default: /workspace/models)",
    )
    args = parser.parse_args()
    download_model(args.model, args.size, args.output)


if __name__ == "__main__":
    main()
