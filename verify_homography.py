"""Verify saved Homography: click image -> floor (X, Y) cm.

Usage:
  python verify_homography.py
  python verify_homography.py --calib calibration/homography.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify Homography by clicking")
    p.add_argument("--calib", default=str(DEFAULT_CALIB), help="homography.json path")
    p.add_argument("--image", default="", help="Override image path (optional)")
    p.add_argument("--max-width", type=int, default=1280)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        raise SystemExit(f"找不到校正檔：{calib_path}")

    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    h_mat = np.array(payload["homography"], dtype=np.float64)
    roi = payload.get("roi") or {}
    image_path = Path(args.image) if args.image else Path(payload["image_path"])
    if not image_path.exists():
        # fallback relative to project
        alt = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
        if alt.exists():
            image_path = alt
        else:
            raise SystemExit(f"找不到影像：{image_path}")

    image = imread_unicode(image_path)
    if image is None:
        raise SystemExit(f"無法讀取影像：{image_path}")

    view, scale = resize_for_preview(image, args.max_width)
    clicks: list[tuple[tuple[float, float], tuple[float, float]]] = []

    win = "Verify Homography"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        full_x = x / scale
        full_y = y / scale
        pts = np.array([[[full_x, full_y]]], dtype=np.float64)
        world = cv2.perspectiveTransform(pts, h_mat)[0, 0]
        wx, wy = float(world[0]), float(world[1])
        clicks.append(((full_x, full_y), (wx, wy)))
        note = ""
        if roi:
            xmin = float(roi.get("x_min_cm", 0))
            ymin = float(roi.get("y_min_cm", 0))
            xmax = float(roi.get("x_max_cm", 0))
            ymax = float(roi.get("y_max_cm", 0))
            if not (xmin <= wx <= xmax and ymin <= wy <= ymax):
                note = "  [超出有效 ROI]"
        print(f"image=({full_x:.1f}, {full_y:.1f}) -> world=({wx:.1f}, {wy:.1f}) cm{note}")

    cv2.setMouseCallback(win, on_mouse)
    print(f"校正檔：{calib_path}")
    print(f"影像：{image_path}")
    print("請只點地板。牆上/螢幕上的點會算出不合理座標。")
    if roi:
        print(
            f"有效 ROI：X {roi.get('x_min_cm')}~{roi.get('x_max_cm')} cm, "
            f"Y {roi.get('y_min_cm')}~{roi.get('y_max_cm')} cm"
        )
    print("左鍵點地板查看平面座標；q 離開。")

    while True:
        canvas = view.copy()
        for (ix, iy), (wx, wy) in clicks[-20:]:
            px, py = int(round(ix * scale)), int(round(iy * scale))
            cv2.circle(canvas, (px, py), 7, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"({wx:.0f},{wy:.0f})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "click floor  [q]uit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
