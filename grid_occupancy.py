"""Map camera click / world (cm) to a floor-grid cell and light it up.

Grid (matches your plan):
  X edges: 0, 35, 80, 125, ..., 530   (first column 35cm, then 45cm)
  Y edges: 0, 45, 90, ..., 540        (all 45cm)

Usage:
  python grid_occupancy.py
  python grid_occupancy.py --x 215 --y 360
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_OUT = Path(__file__).resolve().parent / "test" / "grid_lit_preview.jpg"


def x_edges() -> list[float]:
    # 0, 35, then +45 until 530
    edges = [0.0, 35.0]
    while edges[-1] < 530.0 - 1e-6:
        edges.append(edges[-1] + 45.0)
    # ensure exact end
    if abs(edges[-1] - 530.0) > 1e-6:
        edges[-1] = 530.0
    return edges


def y_edges() -> list[float]:
    edges = [0.0]
    while edges[-1] < 540.0 - 1e-6:
        edges.append(edges[-1] + 45.0)
    if abs(edges[-1] - 540.0) > 1e-6:
        edges[-1] = 540.0
    return edges


X_EDGES = x_edges()
Y_EDGES = y_edges()


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def world_to_cell(x: float, y: float) -> tuple[int, int] | None:
    """Return (col, row) 0-based, or None if outside grid."""
    if x < X_EDGES[0] or x > X_EDGES[-1] or y < Y_EDGES[0] or y > Y_EDGES[-1]:
        return None
    # right/bottom edge belongs to last cell
    col = None
    for i in range(len(X_EDGES) - 1):
        if X_EDGES[i] <= x <= X_EDGES[i + 1] or (
            i == len(X_EDGES) - 2 and abs(x - X_EDGES[i + 1]) < 1e-6
        ):
            if x < X_EDGES[i + 1] or i == len(X_EDGES) - 2:
                col = i
                break
    row = None
    for j in range(len(Y_EDGES) - 1):
        if Y_EDGES[j] <= y <= Y_EDGES[j + 1]:
            if y < Y_EDGES[j + 1] or j == len(Y_EDGES) - 2:
                row = j
                break
    if col is None or row is None:
        return None
    return col, row


def cell_label(col: int, row: int) -> str:
    x0, x1 = X_EDGES[col], X_EDGES[col + 1]
    y0, y1 = Y_EDGES[row], Y_EDGES[row + 1]
    return f"col={col} row={row} | X[{x0:g},{x1:g}) Y[{y0:g},{y1:g})"


def draw_grid(
    active: tuple[int, int] | None,
    valid_x_min: float = 170.0,
    cell_px: int = 48,
) -> np.ndarray:
    n_cols = len(X_EDGES) - 1
    n_rows = len(Y_EDGES) - 1
    margin_l, margin_t = 78, 40
    margin_r, margin_b = 24, 55
    w = margin_l + n_cols * cell_px + margin_r
    h = margin_t + n_rows * cell_px + margin_b
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    for j in range(n_rows):
        for i in range(n_cols):
            x0 = margin_l + i * cell_px
            y0 = margin_t + j * cell_px
            x1 = x0 + cell_px
            y1 = y0 + cell_px
            # dim invalid left columns (X < valid_x_min)
            if X_EDGES[i + 1] <= valid_x_min:
                color = (210, 210, 210)
            elif active is not None and active == (i, j):
                color = (0, 220, 255)  # lit cell (BGR yellow-ish)
            else:
                color = (255, 255, 255)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (180, 80, 180), 1)

    # axis labels (all ticks)
    for i, xv in enumerate(X_EDGES):
        px = margin_l + i * cell_px
        cv2.putText(
            canvas,
            f"{xv:g}",
            (px - 12, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    for j, yv in enumerate(Y_EDGES):
        py = margin_t + j * cell_px
        cv2.putText(
            canvas,
            f"{yv:g}",
            (6, py + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Floor grid (lit = occupied cell)",
        (margin_l, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    if active is not None:
        cv2.putText(
            canvas,
            cell_label(*active),
            (margin_l, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 100, 200),
            1,
            cv2.LINE_AA,
        )
    return canvas


def resize_for_preview(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Light up floor-grid cell from camera/world")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--image", default=str(DEFAULT_IMAGE))
    p.add_argument("--x", type=float, default=None, help="world X cm (skip click)")
    p.add_argument("--y", type=float, default=None, help="world Y cm (skip click)")
    p.add_argument("--valid-xmin", type=float, default=170.0)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max-width", type=int, default=1280)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    active: tuple[int, int] | None = None

    if args.x is not None and args.y is not None:
        active = world_to_cell(args.x, args.y)
        print(f"world=({args.x:g},{args.y:g}) -> {cell_label(*active) if active else 'OUTSIDE'}")
        grid = draw_grid(active, valid_x_min=args.valid_xmin)
        imwrite_unicode(Path(args.out), grid)
        print(f"已輸出：{args.out}")
        cv2.namedWindow("Grid", cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow("Grid", grid)
            if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        return

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    h_mat = np.array(calib["homography"], dtype=np.float64)
    image = imread_unicode(Path(args.image))
    if image is None:
        raise SystemExit(f"無法讀取影像：{args.image}")

    view, scale = resize_for_preview(image, args.max_width)
    last_world = (None, None)

    cam_win = "Camera (click floor)"
    grid_win = "Grid occupancy"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(grid_win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
        nonlocal active, last_world
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        full_x, full_y = x / scale, y / scale
        pts = np.array([[[full_x, full_y]]], dtype=np.float64)
        wx, wy = cv2.perspectiveTransform(pts, h_mat)[0, 0]
        wx, wy = float(wx), float(wy)
        last_world = (wx, wy)
        active = world_to_cell(wx, wy)
        status = cell_label(*active) if active else "OUTSIDE GRID"
        extra = ""
        if active is not None and X_EDGES[active[0] + 1] <= args.valid_xmin:
            extra = " (left desk zone / low confidence)"
        print(f"world=({wx:.1f},{wy:.1f}) -> {status}{extra}")

    cv2.setMouseCallback(cam_win, on_mouse)
    print("左鍵點監視器地板 -> 右側格子會點亮。q 離開。")
    print(f"有效區建議 X>={args.valid_xmin:g} cm")

    while True:
        cam = view.copy()
        if last_world[0] is not None:
            cv2.putText(
                cam,
                f"({last_world[0]:.0f},{last_world[1]:.0f}) cm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        grid = draw_grid(active, valid_x_min=args.valid_xmin)
        cv2.imshow(cam_win, cam)
        cv2.imshow(grid_win, grid)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            imwrite_unicode(Path(args.out), grid)
            print(f"已存：{args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
