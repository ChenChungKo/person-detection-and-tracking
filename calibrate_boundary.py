"""Calibrate floor Homography with conceptual origin TL=(0,0).

You do NOT need to click the real top-left corner if it is occluded.
Only click visible floor tile corners, and enter world coords relative
to the imagined top-left of your ROI.

Example: if the leftmost full-visible tile corner is 35 cm right of your
virtual origin and 90 cm down -> enter: 35 90

Usage:
  python calibrate_boundary.py
  python calibrate_boundary.py --width 530 --height 540
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "homography.json"


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
    p = argparse.ArgumentParser(description="Calibrate with conceptual TL=(0,0)")
    p.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    p.add_argument("--width", type=float, default=530.0, help="ROI width cm")
    p.add_argument("--height", type=float, default=540.0, help="ROI height cm")
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tile-cm", type=float, default=45.0)
    return p.parse_args()


def draw_points(
    view: np.ndarray,
    img_points: list[tuple[float, float]],
    scale: float,
    width_cm: float,
    height_cm: float,
) -> np.ndarray:
    canvas = view.copy()
    for i, (x, y) in enumerate(img_points, start=1):
        px, py = int(round(x * scale)), int(round(y * scale))
        cv2.circle(canvas, (px, py), 8, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 10, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(i),
            (px + 12, py - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    tip = f"points={len(img_points)}  click FLOOR only  [c]ontinue  [u]ndo  [r]eset  [q]uit"
    cv2.putText(canvas, tip, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Virtual origin TL=(0,0) | ROI {width_cm:g}x{height_cm:g} cm (need not be clickable)",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def ask_world_points(n: int, tile_cm: float, width_cm: float, height_cm: float) -> list[tuple[float, float]]:
    print()
    print(f"共 {n} 個影像點。請輸入相對「虛擬左上角 (0,0)」的座標 (cm)。")
    print(f"ROI 範圍約 0~{width_cm:g} , 0~{height_cm:g}（點不到原點沒關係）")
    print(f"例：左側第一可見磚角若在原點右 35cm、下 90cm -> 35 90")
    print(f"地磚參考：{tile_cm:g} cm（左側第一段可為 35）")
    print()
    world: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        while True:
            raw = input(f"點 {i} 的世界座標 X Y: ").strip().replace(",", " ").split()
            if len(raw) != 2:
                print("  請輸入兩個數字，例如：35 90")
                continue
            try:
                world.append((float(raw[0]), float(raw[1])))
                break
            except ValueError:
                print("  數字格式錯誤")
    return world


def make_bird_eye(
    image: np.ndarray,
    h_mat: np.ndarray,
    width_cm: float,
    height_cm: float,
    px_per_cm: float = 2.0,
    margin_cm: float = 20.0,
) -> np.ndarray:
    min_x, max_x = -margin_cm, width_cm + margin_cm
    min_y, max_y = -margin_cm, height_cm + margin_cm
    out_w = max(int(np.ceil((max_x - min_x) * px_per_cm)), 100)
    out_h = max(int(np.ceil((max_y - min_y) * px_per_cm)), 100)
    t = np.array(
        [
            [px_per_cm, 0, -min_x * px_per_cm],
            [0, px_per_cm, -min_y * px_per_cm],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    bird = cv2.warpPerspective(image, t @ h_mat, (out_w, out_h))
    # draw ROI rectangle in bird-eye pixels
    x0 = int(round((0 - min_x) * px_per_cm))
    y0 = int(round((0 - min_y) * px_per_cm))
    x1 = int(round((width_cm - min_x) * px_per_cm))
    y1 = int(round((height_cm - min_y) * px_per_cm))
    cv2.rectangle(bird, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(bird, "(0,0)", (x0 + 5, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return bird


def save_calibration(
    out_path: Path,
    image_path: Path,
    img_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]],
    h_mat: np.ndarray,
    width_cm: float,
    height_cm: float,
    tile_cm: float,
    image_size: tuple[int, int],
    mean_err: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "image_size_wh": [int(image_size[0]), int(image_size[1])],
        "tile_cm": tile_cm,
        "units": "cm",
        "roi": {
            "origin": "top_left_virtual",
            "x_min_cm": 0.0,
            "y_min_cm": 0.0,
            "x_max_cm": width_cm,
            "y_max_cm": height_cm,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "note": "Origin may be outside visible floor; defined conceptually.",
        },
        "image_points_xy": [[float(x), float(y)] for x, y in img_points],
        "world_points_xy": [[float(x), float(y)] for x, y in world_points],
        "homography": h_mat.tolist(),
        "mean_reproj_error_cm": mean_err,
        "note": "Maps image pixels (x,y) to floor plane (X_cm, Y_cm). Virtual TL=(0,0).",
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已儲存：{out_path}")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    image = imread_unicode(image_path)
    if image is None:
        raise SystemExit(f"無法讀取影像：{image_path}")

    h0, w0 = image.shape[:2]
    view, scale = resize_for_preview(image, args.max_width)
    img_points: list[tuple[float, float]] = []

    win = "Calibrate Virtual Origin TL=(0,0)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        full_x, full_y = x / scale, y / scale
        img_points.append((full_x, full_y))
        print(f"點 {len(img_points)}: image=({full_x:.1f}, {full_y:.1f})")

    cv2.setMouseCallback(win, on_mouse)
    print("原點左上角若被擋住：不用點它。")
    print("只點看得到的地板磚角，之後輸入相對虛擬 (0,0) 的 cm 座標。")
    print("建議 8~12 點，左半邊多點一些。按 c 繼續。")

    while True:
        canvas = draw_points(view, img_points, scale, args.width, args.height)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return
        if key == ord("u") and img_points:
            removed = img_points.pop()
            print(f"撤銷：({removed[0]:.1f}, {removed[1]:.1f})")
        elif key == ord("r"):
            img_points.clear()
            print("已重設。")
        elif key == ord("c"):
            if len(img_points) < 4:
                print(f"目前 {len(img_points)} 點，至少要 4 點。")
                continue
            break

    cv2.destroyWindow(win)
    world_points = ask_world_points(len(img_points), args.tile_cm, args.width, args.height)

    src = np.array(img_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)
    h_mat, _ = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        raise SystemExit("Homography 計算失敗。")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    errs = np.linalg.norm(projected - dst, axis=1)
    mean_err = float(errs.mean())
    print("Homography H:")
    print(h_mat)
    print(f"平均重投影誤差：{mean_err:.2f} cm")
    for i, e in enumerate(errs, start=1):
        print(f"  點 {i} 誤差：{float(e):.2f} cm")

    bird = make_bird_eye(image, h_mat, args.width, args.height)
    bird_view, _ = resize_for_preview(bird, args.max_width)
    result_win = "Bird Eye (s=save, q=quit)"
    cv2.namedWindow(result_win, cv2.WINDOW_NORMAL)
    while True:
        show = bird_view.copy()
        cv2.putText(
            show,
            f"mean err {mean_err:.2f} cm  virtual TL=(0,0)  [s]ave  [q]uit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(result_win, show)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            save_calibration(
                Path(args.out),
                image_path,
                img_points,
                world_points,
                h_mat,
                args.width,
                args.height,
                args.tile_cm,
                (w0, h0),
                mean_err,
            )
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
