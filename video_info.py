"""Print basic video metadata (frame count, fps, duration).

Usage:
  python video_info.py
  python video_info.py test/test.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DEFAULT_VIDEO = Path(__file__).resolve().parent / "test" / "test.mp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show video frame count and timing")
    p.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="Video path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.video)
    if not path.exists():
        raise SystemExit(f"找不到影片：{path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_s = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    print(f"檔案：{path}")
    print(f"解析度：{width} x {height}")
    print(f"FPS：{fps:.3f}")
    print(f"總幀數：{frame_count}")
    print(f"長度：{duration_s:.2f} 秒（約 {duration_s / 60:.2f} 分鐘）")


if __name__ == "__main__":
    main()
