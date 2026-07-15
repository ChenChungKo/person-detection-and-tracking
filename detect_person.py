"""Person detection (YOLO) + optional floor mapping with Homography.

Usage:
  python detect_person.py
  python detect_person.py --source test/static_frame.jpg
  python detect_person.py --source test/test.mp4 --no-map
  python detect_person.py --source "rtsp://user:pass@ip:554/stream1"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from latest_frame import LatestFrameCapture

DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_for_preview(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    view = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return view, scale


def load_homography(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["homography"], dtype=np.float64)


def image_to_world(h_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float64)
    world = cv2.perspectiveTransform(pts, h_mat)[0, 0]
    return float(world[0]), float(world[1])


def open_capture(source: str) -> cv2.VideoCapture | None:
    if source.lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        return cap if cap.isOpened() else None
    path = Path(source)
    if path.exists() and path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        return cap if cap.isOpened() else None
    return None


def is_plausible_person_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    frame_w: int,
    min_h_ratio: float = 0.12,
    min_aspect: float = 1.15,
    min_bottom_ratio: float = 0.28,
) -> bool:
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    if bh < min_h_ratio * frame_h:
        return False
    if (bh / bw) < min_aspect:
        return False
    if y2 < min_bottom_ratio * frame_h:
        return False
    if (bw * bh) < 0.005 * frame_w * frame_h:
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO person detection (+ floor mapping)")
    p.add_argument("--source", default=str(DEFAULT_IMAGE), help="image / video / rtsp / webcam index")
    p.add_argument("--calib", default=str(DEFAULT_CALIB), help="homography.json (optional)")
    p.add_argument("--model", default="yolo26s.pt", help="Ultralytics model name/path (e.g. yolo26n/s/m.pt)")
    p.add_argument("--conf", type=float, default=0.45, help="confidence threshold")
    p.add_argument("--min-h-ratio", type=float, default=0.12)
    p.add_argument("--min-aspect", type=float, default=1.15)
    p.add_argument("--min-bottom-ratio", type=float, default=0.28)
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--no-map", action="store_true", help="disable floor mapping overlay")
    return p.parse_args()


def annotate_persons(
    frame: np.ndarray,
    result,
    h_mat: np.ndarray | None,
    conf_thres: float,
    min_h_ratio: float,
    min_aspect: float,
    min_bottom_ratio: float,
) -> np.ndarray:
    out = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    fh, fw = frame.shape[:2]
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        # COCO class 0 = person
        if cls_id != 0:
            continue
        conf = float(boxes.conf[i].item())
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        if not is_plausible_person_box(
            x1, y1, x2, y2, fh, fw, min_h_ratio, min_aspect, min_bottom_ratio
        ):
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        foot_x = 0.5 * (x1 + x2)
        foot_y = float(y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out, (int(foot_x), int(foot_y)), 6, (0, 0, 255), -1)

        label = f"person {conf:.2f}"
        if h_mat is not None:
            wx, wy = image_to_world(h_mat, foot_x, foot_y)
            label = f"person {conf:.2f} | ({wx:.0f},{wy:.0f})cm"
            print(f"person conf={conf:.2f} foot=({foot_x:.1f},{foot_y:.1f}) world=({wx:.1f},{wy:.1f}) cm")

        cv2.putText(
            out,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    h_mat = None if args.no_map else load_homography(Path(args.calib))
    if h_mat is None and not args.no_map:
        print("未載入 Homography，只做偵測不映射。")

    source = args.source
    src_path = Path(source)
    is_image = src_path.exists() and src_path.suffix.lower() in IMAGE_EXTS

    win = "Person Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"偵測：YOLO（{args.model}），conf={args.conf}。按 q 結束。")

    if is_image:
        frame = imread_unicode(src_path)
        if frame is None:
            raise SystemExit(f"無法讀取影像：{source}")
        results = model.predict(frame, conf=args.conf, classes=[0], verbose=False)
        annotated = annotate_persons(
            frame,
            results[0],
            h_mat,
            args.conf,
            args.min_h_ratio,
            args.min_aspect,
            args.min_bottom_ratio,
        )
        view, _ = resize_for_preview(annotated, args.max_width)
        while True:
            cv2.imshow(win, view)
            if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        return

    cap = open_capture(source)
    if cap is None or not cap.isOpened():
        raise SystemExit(f"無法開啟來源：{source}")

    use_latest = source.lower().startswith("rtsp://")
    reader: LatestFrameCapture | None = LatestFrameCapture(cap) if use_latest else None
    if use_latest:
        print("RTSP：啟用最新幀讀取（推論慢時丟棄舊幀，降低延遲感）")
        for _ in range(50):
            ok, frame = reader.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        else:
            reader.release()
            raise SystemExit("RTSP 連線後未收到畫面。")

    try:
        while True:
            if reader is not None:
                ok, frame = reader.read()
                if not ok or frame is None:
                    if not reader.is_alive():
                        print("讀取結束或失敗。")
                        break
                    time.sleep(0.01)
                    continue
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("讀取結束或失敗。")
                    break
            results = model.predict(frame, conf=args.conf, classes=[0], verbose=False)
            annotated = annotate_persons(
                frame,
                results[0],
                h_mat,
                args.conf,
                args.min_h_ratio,
                args.min_aspect,
                args.min_bottom_ratio,
            )
            view, _ = resize_for_preview(annotated, args.max_width)
            cv2.imshow(win, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if reader is not None:
            reader.release()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
