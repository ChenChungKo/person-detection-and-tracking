"""Official Ultralytics multi-object track preview (docs default).

Uses YOLO26 + BoT-SORT (Ultralytics default tracker), no Stable-ID / OSNet.
Purpose: check whether multiple people get boxes and short-term IDs.

  python track_preview.py --source "rtsp://user:pass@ip:554/stream1"
  python track_preview.py --source test/test2.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultralytics official track preview")
    p.add_argument(
        "--source",
        default="rtsp://oriongo:123456789@192.168.0.200:554/stream1",
        help="RTSP URL or video path",
    )
    p.add_argument("--model", default="yolo26s.pt", help="YOLO weights (docs use yolo26n.pt)")
    p.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Ultralytics tracker yaml (default botsort.yaml = docs default)",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--all-classes",
        action="store_true",
        help="track every class (default: person only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.source).startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    from ultralytics import YOLO

    model = YOLO(args.model)
    kw = dict(
        source=args.source,
        tracker=args.tracker,
        persist=True,
        show=True,
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=True,
    )
    if not args.all_classes:
        kw["classes"] = [0]

    print(
        f"官方 track：{args.model} + {args.tracker}（BoT-SORT 為文件預設）"
        f"{'' if args.all_classes else '，只抓 person'}"
    )
    print("視窗出現後按 q 結束。")
    for _result in model.track(**kw):
        pass


if __name__ == "__main__":
    main()
