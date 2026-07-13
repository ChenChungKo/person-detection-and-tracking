"""Interactive floor-plane Homography calibration.

Usage (with .venv active):
  python calibrate_homography.py
  python calibrate_homography.py "test/static_frame.jpg"

Controls (point picking):
  Left click  - add image point
  u           - undo last point
  c           - continue (>= 4 points) and enter world coords in terminal
  r           - reset all points
  q / ESC     - quit

After Homography is computed:
  s           - save calibration JSON
  q / ESC     - quit
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
    """cv2.imread fails on non-ASCII Windows paths; decode via NumPy instead."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Floor Homography calibrator")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="Static camera image path",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Preview max width (clicks map back to full resolution)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output JSON path",
    )
    parser.add_argument(
        "--tile-cm",
        type=float,
        default=45.0,
        help="Floor tile size in cm (for reference only)",
    )
    return parser.parse_args()


def resize_for_preview(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    view = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return view, scale


def draw_points(view: np.ndarray, img_points: list[tuple[float, float]], scale: float) -> np.ndarray:
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
    return canvas


def ask_world_points(n: int, tile_cm: float) -> list[tuple[float, float]]:
    print()
    print(f"共 {n} 個影像點。請依序輸入每個點的平面座標 (單位 cm)。")
    print(f"地磚參考尺寸：{tile_cm:g} cm（例如往右 2 格、往下 3 格 -> 90 135）")
    print("格式：X_cm Y_cm   例：90 135")
    print("座標系請自行定義（例如紅框左上角為原點，即使原點點不到也可以）。")
    print()

    world: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        while True:
            raw = input(f"點 {i} 的世界座標 X Y: ").strip()
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print("  請輸入兩個數字，例如：90 135")
                continue
            try:
                x_cm = float(parts[0])
                y_cm = float(parts[1])
            except ValueError:
                print("  數字格式錯誤，請再試一次。")
                continue
            world.append((x_cm, y_cm))
            break
    return world


def compute_homography(
    img_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    src = np.array(img_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)
    h_mat, status = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        raise RuntimeError("Homography 計算失敗，請檢查點位是否共線或輸入錯誤。")
    return h_mat, status


def make_bird_eye(
    image: np.ndarray,
    h_mat: np.ndarray,
    world_points: list[tuple[float, float]],
    px_per_cm: float = 2.0,
    margin_cm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [p[0] for p in world_points]
    ys = [p[1] for p in world_points]
    min_x, max_x = min(xs) - margin_cm, max(xs) + margin_cm
    min_y, max_y = min(ys) - margin_cm, max(ys) + margin_cm

    out_w = int(np.ceil((max_x - min_x) * px_per_cm))
    out_h = int(np.ceil((max_y - min_y) * px_per_cm))
    out_w = max(out_w, 100)
    out_h = max(out_h, 100)

    # Map world(cm) -> bird-eye pixels
    t = np.array(
        [
            [px_per_cm, 0, -min_x * px_per_cm],
            [0, px_per_cm, -min_y * px_per_cm],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    warp_m = t @ h_mat
    bird = cv2.warpPerspective(image, warp_m, (out_w, out_h))
    return bird, warp_m


def save_calibration(
    out_path: Path,
    image_path: Path,
    img_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]],
    h_mat: np.ndarray,
    tile_cm: float,
    image_size: tuple[int, int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "image_size_wh": [int(image_size[0]), int(image_size[1])],
        "tile_cm": tile_cm,
        "units": "cm",
        "image_points_xy": [[float(x), float(y)] for x, y in img_points],
        "world_points_xy": [[float(x), float(y)] for x, y in world_points],
        "homography": h_mat.tolist(),
        "note": "Maps image pixels (x,y) to floor plane (X_cm, Y_cm).",
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已儲存：{out_path}")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"找不到影像：{image_path}")

    image = imread_unicode(image_path)
    if image is None:
        raise SystemExit(f"無法讀取影像：{image_path}")

    h0, w0 = image.shape[:2]
    view, scale = resize_for_preview(image, args.max_width)
    img_points: list[tuple[float, float]] = []

    win = "Homography Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map preview click -> full-resolution image coordinates
            full_x = x / scale
            full_y = y / scale
            img_points.append((full_x, full_y))
            print(f"點 {len(img_points)}: image=({full_x:.1f}, {full_y:.1f})")

    cv2.setMouseCallback(win, on_mouse)

    print(f"影像：{image_path} ({w0}x{h0})")
    print("在地板上點選對應點（至少 4 點）。原點看不見也不必點到。")
    print("按 c 繼續輸入世界座標；u 撤銷；r 重設；q 離開。")

    while True:
        canvas = draw_points(view, img_points, scale)
        tip = f"points={len(img_points)}  [c]ontinue  [u]ndo  [r]eset  [q]uit"
        cv2.putText(
            canvas,
            tip,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return
        if key == ord("u"):
            if img_points:
                removed = img_points.pop()
                print(f"撤銷：({removed[0]:.1f}, {removed[1]:.1f})")
        elif key == ord("r"):
            img_points.clear()
            print("已重設所有點。")
        elif key == ord("c"):
            if len(img_points) < 4:
                print(f"目前只有 {len(img_points)} 點，至少需要 4 點。")
                continue
            break

    cv2.destroyWindow(win)
    world_points = ask_world_points(len(img_points), args.tile_cm)
    h_mat, _status = compute_homography(img_points, world_points)

    print()
    print("Homography matrix H:")
    print(h_mat)

    # Reprojection error (cm)
    src = np.array(img_points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, h_mat).reshape(-1, 2)
    errs = np.linalg.norm(projected - np.array(world_points, dtype=np.float64), axis=1)
    print(f"平均重投影誤差：{float(errs.mean()):.2f} cm（越小越好）")

    bird, _ = make_bird_eye(image, h_mat, world_points)
    bird_view, _ = resize_for_preview(bird, args.max_width)

    result_win = "Bird Eye Preview (press s to save, q to quit)"
    cv2.namedWindow(result_win, cv2.WINDOW_NORMAL)
    while True:
        show = bird_view.copy()
        cv2.putText(
            show,
            f"mean err {float(errs.mean()):.2f} cm  [s]ave  [q]uit",
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
                args.tile_cm,
                (w0, h0),
            )
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
