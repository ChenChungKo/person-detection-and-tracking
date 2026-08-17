"""Manually pick four floor marks (A B C O) on a camera frame.

Click visible floor only. World cm come from the current Homography.

Usage:
  python pick_floor_marks.py
  python pick_floor_marks.py --source test/static_frame.jpg
  python pick_floor_marks.py --source test/test4.mp4 --frame 1

Keys:
  left-click  place next mark (A → B → C → O)
  u           undo last mark
  s           save and exit (needs all four)
  r           clear all
  q           quit without saving
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from grid_occupancy import MARK_COLORS_RGB, MARK_NAMES, imread_unicode, landmark_bgr

DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "floor_marks.json"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"


def load_homography(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["homography"], dtype=np.float64)


def image_to_world(h_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float64)
    world = cv2.perspectiveTransform(pts, h_mat)[0, 0]
    return float(world[0]), float(world[1])


def open_frame(source: str, frame_idx: int) -> tuple[np.ndarray, str]:
    path = Path(source)
    if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = imread_unicode(path)
        if img is None:
            raise SystemExit(f"無法讀取影像：{path}")
        return img, str(path)
    if path.exists() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"無法開啟影片：{path}")
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise SystemExit(f"無法讀取第 {frame_idx} 幀：{path}")
        return frame, f"{path}#frame={frame_idx}"
    raise SystemExit(f"找不到來源：{source}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick four floor marks A B C O")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--source", default=str(DEFAULT_IMAGE))
    p.add_argument("--frame", type=int, default=1, help="video frame index (0-based)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max-width", type=int, default=1280)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        raise SystemExit(f"找不到校正檔：{calib_path}")
    h_mat = load_homography(calib_path)
    image, source_label = open_frame(args.source, args.frame)

    h, w = image.shape[:2]
    scale = 1.0
    view = image
    if args.max_width > 0 and w > args.max_width:
        scale = args.max_width / float(w)
        view = cv2.resize(image, (args.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    # name -> (wx, wy, full_ix, full_iy)
    picks: list[tuple[str, float, float, float, float]] = []
    win = "Pick Floor Marks (A B C O)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(picks) >= len(MARK_NAMES):
            print("已選滿 4 點。按 s 存檔，或 u 撤銷／r 重選。")
            return
        full_x = x / scale
        full_y = y / scale
        wx, wy = image_to_world(h_mat, full_x, full_y)
        name = MARK_NAMES[len(picks)]
        picks.append((name, wx, wy, full_x, full_y))
        print(f"{name}: image=({full_x:.1f},{full_y:.1f}) -> world=({wx:.1f},{wy:.1f}) cm")
        if len(picks) == len(MARK_NAMES):
            print("四點齊了。按 s 存檔，u 撤銷最後一點，r 重來，q 離開不存。")

    cv2.setMouseCallback(win, on_mouse)
    print(f"校正：{calib_path}")
    print(f"來源：{source_label}")
    print("請依序點地板：A（近左）→ B（遠左）→ C（近右）→ O（正中心）")
    print("只點看得見的地面，不要點桌子／人／牆。")
    print("按鍵：u 撤銷  r 重選  s 存檔  q 離開")

    out_path = Path(args.out)
    while True:
        canvas = view.copy()
        for name, wx, wy, ix, iy in picks:
            rgb = MARK_COLORS_RGB[name]
            bgr = landmark_bgr(rgb)
            px, py = int(round(ix * scale)), int(round(iy * scale))
            cv2.circle(canvas, (px, py), 14, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 11, bgr, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{name} ({wx:.0f},{wy:.0f})",
                (px + 16, py + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{name} ({wx:.0f},{wy:.0f})",
                (px + 16, py + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                bgr,
                2,
                cv2.LINE_AA,
            )
        nxt = MARK_NAMES[len(picks)] if len(picks) < len(MARK_NAMES) else "done — press s"
        tip = f"next: {nxt}   [u]ndo [r]eset [s]ave [q]uit"
        cv2.putText(canvas, tip, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("已離開，未存檔。")
            break
        if key == ord("u"):
            if picks:
                removed = picks.pop()
                print(f"撤銷 {removed[0]}")
        elif key == ord("r"):
            picks.clear()
            print("已清空，請重新從 A 開始點。")
        elif key == ord("s"):
            if len(picks) != len(MARK_NAMES):
                print(f"還缺 {len(MARK_NAMES) - len(picks)} 點，無法存檔。")
                continue
            marks = []
            for name, wx, wy, ix, iy in picks:
                marks.append(
                    {
                        "name": name,
                        "world_xy_cm": [round(wx, 2), round(wy, 2)],
                        "image_xy": [round(ix, 2), round(iy, 2)],
                        "rgb": list(MARK_COLORS_RGB[name]),
                    }
                )
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "calib": str(calib_path).replace("\\", "/"),
                "source": source_label.replace("\\", "/"),
                "image_size_wh": [int(w), int(h)],
                "marks": marks,
                "note": "Manual floor marks for report overlay (A B C O).",
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"已存：{out_path}")
            for m in marks:
                print(f"  {m['name']}: world={m['world_xy_cm']}  image={m['image_xy']}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
