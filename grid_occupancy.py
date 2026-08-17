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
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_OUT = Path(__file__).resolve().parent / "test" / "grid_lit_preview.jpg"

# Windows TTF for crisp labels (OpenCV putText is blurry on many displays)
_FONT_REGULAR: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
_FONT_BOLD: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None


def _load_fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    global _FONT_REGULAR, _FONT_BOLD
    if _FONT_REGULAR is not None and _FONT_BOLD is not None:
        return _FONT_REGULAR, _FONT_BOLD
    # Prefer CJK fonts first so Traditional Chinese titles render (not □□□)
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),      # Microsoft YaHei
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
        Path(r"C:\Windows\Fonts\mingliub.ttc"),
        Path(r"C:\Windows\Fonts\kaiu.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                _FONT_REGULAR = ImageFont.truetype(str(path), 18)
                _FONT_BOLD = ImageFont.truetype(str(path), 22)
                return _FONT_REGULAR, _FONT_BOLD
            except OSError:
                continue
    _FONT_REGULAR = ImageFont.load_default()
    _FONT_BOLD = _FONT_REGULAR
    return _FONT_REGULAR, _FONT_BOLD


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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

# Report overlay marks (world cm). No far-right point.
MARK_NAMES: tuple[str, ...] = ("A", "B", "C", "O")
MARK_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "A": (255, 80, 80),
    "B": (40, 160, 255),
    "C": (40, 200, 90),
    "O": (255, 230, 0),
}
DEFAULT_FLOOR_MARKS_PATH = Path(__file__).resolve().parent / "calibration" / "floor_marks.json"
# A near-left, B far-left, C near-right, O aisle center.
_DEFAULT_MARK_XY: dict[str, tuple[float, float]] = {
    "A": (170.0, 450.0),
    "B": (170.0, 180.0),
    "C": (440.0, 450.0),
    "O": (260.0, 315.0),
}


def _default_floor_marks() -> tuple[tuple[str, float, float, tuple[int, int, int]], ...]:
    return tuple(
        (name, _DEFAULT_MARK_XY[name][0], _DEFAULT_MARK_XY[name][1], MARK_COLORS_RGB[name])
        for name in MARK_NAMES
    )


def load_floor_marks(
    path: Path | None = None,
) -> tuple[tuple[str, float, float, tuple[int, int, int]], ...]:
    marks_path = path or DEFAULT_FLOOR_MARKS_PATH
    if not marks_path.exists():
        return _default_floor_marks()
    try:
        payload = json.loads(marks_path.read_text(encoding="utf-8"))
        rows = payload.get("marks") or []
        by_name: dict[str, tuple[float, float, tuple[int, int, int]]] = {}
        for row in rows:
            name = str(row["name"]).upper()
            if name not in MARK_COLORS_RGB:
                continue
            wx, wy = row["world_xy_cm"]
            rgb = tuple(int(v) for v in row.get("rgb", MARK_COLORS_RGB[name]))
            if len(rgb) != 3:
                rgb = MARK_COLORS_RGB[name]
            by_name[name] = (float(wx), float(wy), (rgb[0], rgb[1], rgb[2]))
        ordered: list[tuple[str, float, float, tuple[int, int, int]]] = []
        for name in MARK_NAMES:
            if name not in by_name:
                raise KeyError(name)
            wx, wy, rgb = by_name[name]
            ordered.append((name, wx, wy, rgb))
        return tuple(ordered)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _default_floor_marks()


FLOOR_MARKS = load_floor_marks()
FLOOR_LANDMARKS = tuple(m for m in FLOOR_MARKS if m[0] != "O")
GRID_MARGIN_L = 96
GRID_MARGIN_T = 52
GRID_CELL_PX = 72


def world_to_grid_px(
    wx: float,
    wy: float,
    cell_px: int = GRID_CELL_PX,
) -> tuple[int, int]:
    """Map world cm to pixel on the drawn bird-eye grid."""

    def axis_pos(value: float, edges: list[float]) -> float:
        if value <= edges[0]:
            return 0.0
        if value >= edges[-1]:
            return float(len(edges) - 1)
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo <= value <= hi:
                span = hi - lo
                frac = 0.0 if span <= 1e-9 else (value - lo) / span
                return i + frac
        return float(len(edges) - 1)

    px = GRID_MARGIN_L + axis_pos(wx, X_EDGES) * cell_px
    py = GRID_MARGIN_T + axis_pos(wy, Y_EDGES) * cell_px
    return int(round(px)), int(round(py))


def landmark_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (int(b), int(g), int(r))


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


def world_to_cell(
    x: float,
    y: float,
    margin_cm: float = 0.0,
) -> tuple[int, int] | None:
    """Return (col, row) 0-based, or None if outside grid.

    ``margin_cm``: if the point is only slightly outside the rectangle
    [0,530]×[0,540] (Homography / foot noise near edges), clamp into the
    nearest in-grid coordinate and still return a cell. Far outliers stay OUT.
    """
    x0, x1 = X_EDGES[0], X_EDGES[-1]
    y0, y1 = Y_EDGES[0], Y_EDGES[-1]
    m = max(0.0, float(margin_cm))
    if x < x0 - m or x > x1 + m or y < y0 - m or y > y1 + m:
        return None
    # Soft edge: snap barely-outside points onto the grid boundary interior.
    eps = 1e-3
    x = min(max(x, x0), x1 - eps)
    y = min(max(y, y0), y1 - eps)

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


def id_fill_color(track_id: int) -> tuple[int, int, int]:
    """Stable pastel fill per person ID (RGB for Pillow)."""
    palette = (
        (255, 220, 0),
        (120, 200, 255),
        (255, 160, 180),
        (160, 230, 160),
        (220, 180, 255),
        (255, 200, 120),
        (180, 230, 220),
        (255, 190, 140),
    )
    return palette[(max(1, int(track_id)) - 1) % len(palette)]


def id_bgr_color(track_id: int) -> tuple[int, int, int]:
    """Same ID color as the floor grid, in OpenCV BGR."""
    r, g, b = id_fill_color(track_id)
    return (int(b), int(g), int(r))


# Cached empty board (axes + title) so 2-person redraws stay cheap.
_EMPTY_GRID_CACHE: dict[tuple[float, int], Image.Image] = {}


def _empty_grid_image(valid_x_min: float, cell_px: int) -> Image.Image:
    key = (float(valid_x_min), int(cell_px))
    cached = _EMPTY_GRID_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    font, font_title = _load_fonts()
    n_cols = len(X_EDGES) - 1
    n_rows = len(Y_EDGES) - 1
    margin_l, margin_t = 96, 52
    margin_r, margin_b = 28, 64
    w = margin_l + n_cols * cell_px + margin_r
    h = margin_t + n_rows * cell_px + margin_b

    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    purple = (140, 60, 160)
    desk = (210, 210, 210)
    white = (255, 255, 255)

    for j in range(n_rows):
        for i in range(n_cols):
            x0 = margin_l + i * cell_px
            y0 = margin_t + j * cell_px
            fill = (
                desk
                if valid_x_min > 0 and X_EDGES[i + 1] <= valid_x_min
                else white
            )
            draw.rectangle(
                (x0, y0, x0 + cell_px - 1, y0 + cell_px - 1),
                fill=fill,
                outline=purple,
                width=2,
            )

    for i, xv in enumerate(X_EDGES):
        px = margin_l + i * cell_px
        draw.text((px - 14, h - margin_b + 8), f"{xv:g}", fill=(40, 40, 40), font=font)
    for j, yv in enumerate(Y_EDGES):
        py = margin_t + j * cell_px
        draw.text((8, py - 10), f"{yv:g}", fill=(40, 40, 40), font=font)

    draw.text(
        (margin_l, 8),
        "地板格子（色格＝有人，格內＝ID）",
        fill=(20, 20, 20),
        font=font_title,
    )
    if valid_x_min > 0:
        draw.text(
            (margin_l, 30),
            f"淺灰＝桌區／低可信（X < {valid_x_min:g} cm）",
            fill=(90, 90, 90),
            font=font,
        )
    else:
        draw.text(
            (margin_l, 30),
            "全格可用（未標記桌區低可信）",
            fill=(90, 90, 90),
            font=font,
        )
    _EMPTY_GRID_CACHE[key] = img
    return img.copy()


def _draw_grid_landmarks(draw, cell_px: int) -> None:
    """Same four floor marks as the camera overlay: A, B, C, and center O."""
    try:
        font_mark = ImageFont.truetype(str(Path(r"C:\Windows\Fonts\msyhbd.ttc")), 28)
    except OSError:
        font_mark = _load_fonts()[1]
    radius = 14
    for name, wx, wy, rgb in FLOOR_MARKS:
        px, py = world_to_grid_px(wx, wy, cell_px)
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=rgb,
            outline=(20, 20, 20),
            width=3,
        )
        draw.text((px + 16, py - 20), name, fill=rgb, font=font_mark)


def draw_grid(
    active: tuple[int, int] | None,
    valid_x_min: float = 0.0,
    cell_px: int = 72,
    occupancy: dict[tuple[int, int], list[int]] | None = None,
    landmarks: bool = False,
) -> np.ndarray:
    """Draw grid with Pillow (sharp text/lines on Windows).

    ``occupancy`` maps (col, row) -> list of track IDs to show in-cell
    (e.g. who is standing where). When set, those cells are lit and labeled.
    """
    font, font_title = _load_fonts()
    try:
        font_id = ImageFont.truetype(str(Path(r"C:\Windows\Fonts\msyhbd.ttc")), 20)
    except OSError:
        font_id = font_title
    n_cols = len(X_EDGES) - 1
    n_rows = len(Y_EDGES) - 1
    margin_l, margin_t = 96, 52
    margin_r, margin_b = 28, 64
    w = margin_l + n_cols * cell_px + margin_r
    h = margin_t + n_rows * cell_px + margin_b

    img = _empty_grid_image(valid_x_min, cell_px)
    draw = ImageDraw.Draw(img)

    purple = (140, 60, 160)
    lit = (255, 220, 0)  # click / active preview only (no person ID yet)
    occ = occupancy or {}

    # Only repaint occupied / active cells (cheap when 1–2 people).
    cells_to_paint: set[tuple[int, int]] = set(occ)
    if active is not None:
        cells_to_paint.add(active)
    for i, j in cells_to_paint:
        if not (0 <= i < n_cols and 0 <= j < n_rows):
            continue
        x0 = margin_l + i * cell_px
        y0 = margin_t + j * cell_px
        ids = sorted({int(t) for t in occ.get((i, j), [])})
        if ids:
            fill = id_fill_color(ids[0])
        elif active is not None and active == (i, j):
            # Manual click preview — not a tracked person.
            fill = lit
        else:
            # Occupied but no ID yet: do not paint yellow (avoids ID flash).
            continue
        draw.rectangle(
            (x0, y0, x0 + cell_px - 1, y0 + cell_px - 1),
            fill=fill,
            outline=purple,
            width=2,
        )
        if ids:
            label = "\n".join(f"ID{t}" for t in ids[:3])
            if len(ids) > 3:
                label += f"\n+{len(ids) - 3}"
            tx = x0 + 8
            ty = y0 + max(4, (cell_px - 22 * min(len(ids), 3)) // 2)
            draw.text((tx, ty), label, fill=(20, 20, 20), font=font_id)
    parts = [
        f"ID{tid}@({c},{r})"
        for (c, r), ids in sorted(occ.items())
        for tid in sorted(ids)
    ]
    if parts:
        summary = "  ".join(parts[:8])
        if len(parts) > 8:
            summary += f"  +{len(parts) - 8}"
        draw.text((margin_l, h - 28), summary, fill=(200, 100, 0), font=font)
    elif active is not None:
        draw.text(
            (margin_l, h - 28),
            cell_label(*active),
            fill=(200, 100, 0),
            font=font,
        )
    if landmarks:
        _draw_grid_landmarks(draw, cell_px)
    return _pil_to_bgr(img)


def show_grid_window(win: str, grid_bgr: np.ndarray) -> bool:
    """Show grid at 1:1 pixel size to avoid OpenCV upscale blur."""
    return show_fixed_window(win, grid_bgr)


def show_fixed_window(win: str, image_bgr: np.ndarray) -> bool:
    """Show image with window size locked to the image pixel size.

    Returns False if the window was closed (so callers can exit cleanly).
    """
    h, w = image_bgr.shape[:2]
    if not hasattr(show_fixed_window, "_ready"):
        show_fixed_window._ready = set()  # type: ignore[attr-defined]
    ready: set[str] = show_fixed_window._ready  # type: ignore[attr-defined]
    try:
        if win not in ready:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            ready.add(win)
        # Keep size fixed every frame (matches preview pixels; ignores manual drag)
        cv2.resizeWindow(win, w, h)
        cv2.imshow(win, image_bgr)
        # Closed via title-bar X → property becomes negative.
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            ready.discard(win)
            return False
    except cv2.error:
        ready.discard(win)
        return False
    return True


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
    p.add_argument(
        "--valid-xmin",
        type=float,
        default=0.0,
        help="desk-zone gray mask threshold in cm; 0 = full grid (no gray); 170 = old desk mask",
    )
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
        show_grid_window("Grid", grid)
        while True:
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
        if active is not None and args.valid_xmin > 0 and X_EDGES[active[0] + 1] <= args.valid_xmin:
            extra = " (left desk zone / low confidence)"
        print(f"world=({wx:.1f},{wy:.1f}) -> {status}{extra}")

    cv2.setMouseCallback(cam_win, on_mouse)
    print("左鍵點監視器地板 -> 右側格子會點亮。q 離開。")
    if args.valid_xmin > 0:
        print(f"有效區建議 X>={args.valid_xmin:g} cm（左側淺灰為桌區低可信）")
    else:
        print("全格可用（--valid-xmin 0，未標記桌區）")
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
        show_grid_window(grid_win, grid)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            imwrite_unicode(Path(args.out), grid)
            print(f"已存：{args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
