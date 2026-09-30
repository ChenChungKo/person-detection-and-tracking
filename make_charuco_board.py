"""Printable ChArUco board for floor calibration, tiled onto A3 at actual size.

Defaults match the existing floor chessboard footprint:
  5 x 4 squares, 20 cm each, ArUco marker 14 cm (0.7 x square).
  Dictionary: DICT_4X4_50. Print at 100% / actual size, not "fit to page".
  Trim the white page margin, then align the small overlap and tape.

Usage:
  python make_charuco_board.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from make_chessboard_board import (
    PAPER_SIZES_CM,
    _font,
    make_assembly_guide,
)

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "charuco_print"
SQUARES_X = 5
SQUARES_Y = 4
SQUARE_CM = 20.0
MARKER_CM = 14.0
QUIET_MARGIN_CM = 2.0
PAPER = "a3"
PAGE_MARGIN_CM = 0.5
OVERLAP_CM = 0.3
DPI = 300
DICTIONARY = "DICT_4X4_50"


def build_charuco_board(square_px: int, quiet_margin_px: int) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_CM / 100.0,
        MARKER_CM / 100.0,
        dictionary,
    )
    pattern = board.generateImage((SQUARES_X * square_px, SQUARES_Y * square_px))
    if pattern.ndim == 2:
        pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(pattern)
    canvas = Image.new(
        "RGB",
        (image.width + 2 * quiet_margin_px, image.height + 2 * quiet_margin_px),
        (255, 255, 255),
    )
    canvas.paste(image, (quiet_margin_px, quiet_margin_px))
    return canvas


def main() -> None:
    out_dir = DEFAULT_OUT
    if out_dir.exists():
        for stale in out_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    ppc = DPI / 2.54
    square_px = int(round(SQUARE_CM * ppc))
    quiet_margin_px = int(round(QUIET_MARGIN_CM * ppc))
    board = build_charuco_board(square_px, quiet_margin_px)
    board_w_cm = SQUARES_X * SQUARE_CM + 2 * QUIET_MARGIN_CM
    board_h_cm = SQUARES_Y * SQUARE_CM + 2 * QUIET_MARGIN_CM

    board_path = out_dir / "board_full_preview.png"
    board.save(board_path)

    portrait_w_cm, portrait_h_cm = PAPER_SIZES_CM[PAPER]

    def page_count(page_w_cm: float, page_h_cm: float) -> tuple[int, int]:
        usable_w = page_w_cm - 2 * PAGE_MARGIN_CM
        usable_h = page_h_cm - 2 * PAGE_MARGIN_CM
        best: tuple[int, int] | None = None
        for rows in range(1, 8):
            for cols in range(1, 8):
                crop_w = (board_w_cm + (cols - 1) * OVERLAP_CM) / cols
                crop_h = (board_h_cm + (rows - 1) * OVERLAP_CM) / rows
                if crop_w <= usable_w + 1e-6 and crop_h <= usable_h + 1e-6:
                    if best is None or rows * cols < best[0] * best[1]:
                        best = (rows, cols)
        return best if best is not None else (1, 1)

    rows_p, cols_p = page_count(portrait_w_cm, portrait_h_cm)
    rows_l, cols_l = page_count(portrait_h_cm, portrait_w_cm)
    if rows_l * cols_l < rows_p * cols_p:
        page_w_cm, page_h_cm = portrait_h_cm, portrait_w_cm
        n_rows, n_cols = rows_l, cols_l
    else:
        page_w_cm, page_h_cm = portrait_w_cm, portrait_h_cm
        n_rows, n_cols = rows_p, cols_p

    crop_w_cm = (board_w_cm + (n_cols - 1) * OVERLAP_CM) / n_cols
    crop_h_cm = (board_h_cm + (n_rows - 1) * OVERLAP_CM) / n_rows
    crop_w_px = int(round(crop_w_cm * ppc))
    crop_h_px = int(round(crop_h_cm * ppc))
    step_w_px = int(round((crop_w_cm - OVERLAP_CM) * ppc))
    step_h_px = int(round((crop_h_cm - OVERLAP_CM) * ppc))
    page_w_px = int(round(page_w_cm * ppc))
    page_h_px = int(round(page_h_cm * ppc))

    pages: list[Image.Image] = []
    tiles: list[dict] = []
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * step_w_px
            y0 = row * step_h_px
            x1 = x0 + crop_w_px
            y1 = y0 + crop_h_px
            if col == n_cols - 1:
                x1 = board.width
                x0 = max(0, x1 - crop_w_px)
            if row == n_rows - 1:
                y1 = board.height
                y0 = max(0, y1 - crop_h_px)
            x0 = max(0, min(x0, board.width - 1))
            y0 = max(0, min(y0, board.height - 1))
            x1 = max(x0 + 1, min(x1, board.width))
            y1 = max(y0 + 1, min(y1, board.height))

            crop = board.crop((x0, y0, x1, y1))
            page = Image.new("RGB", (page_w_px, page_h_px), color=(255, 255, 255))
            ox = (page_w_px - crop.width) // 2
            oy = (page_h_px - crop.height) // 2
            page.paste(crop, (ox, oy))
            label = f"R{row + 1}C{col + 1}"
            if oy > 40:
                draw = ImageDraw.Draw(page)
                draw.text((ox + 8, 8), label, fill=(180, 0, 0), font=_font(28))
            page_name = f"page_{label}.png"
            page.save(out_dir / page_name)
            pages.append(page)
            tiles.append({"label": label, "crop_px": (x0, y0, x1, y1), "file": page_name})

    pdf_path = out_dir / "charuco_pages.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=DPI)
    guide = make_assembly_guide(board, tiles)
    guide_path = out_dir / "assembly_guide.png"
    guide.save(guide_path)

    spec = {
        "dictionary": DICTIONARY,
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_cm": SQUARE_CM,
        "marker_cm": MARKER_CM,
        "quiet_margin_cm": QUIET_MARGIN_CM,
        "board_cm": [board_w_cm, board_h_cm],
        "paper": PAPER.upper(),
        "dpi": DPI,
        "pages": f"{n_rows}x{n_cols}",
        "page_margin_cm": PAGE_MARGIN_CM,
        "overlap_cm": OVERLAP_CM,
        "assembly": "trim the white page margin, then align the overlap and tape",
    }
    (out_dir / "charuco_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )

    orient = "橫向" if page_w_cm > page_h_cm else "縱向"
    print(f"ChArUco：{SQUARES_X}x{SQUARES_Y} 格，每格 {SQUARE_CM:g} cm，標記 {MARKER_CM:g} cm")
    print(f"整塊約 {board_w_cm:g} x {board_h_cm:g} cm（含白邊）")
    print(f"字典：{DICTIONARY}")
    print(
        f"拼頁：{PAPER.upper()} {orient}，{n_rows} 排 x {n_cols} 欄 = {n_rows * n_cols} 頁"
    )
    print(f"PDF：{pdf_path}")


if __name__ == "__main__":
    main()
