"""Generate a print-and-tile physical chessboard for floor Homography calibration.

Produces:
  - A full-resolution board preview PNG (for reference).
  - An assembly-guide PNG showing how the printed pages tile together.
  - One PNG + one multi-page PDF per printed page, sized so printing at
    "100% / actual size" (NOT "fit to page") reproduces the exact physical
    square size in --square-cm. Tape the pages together, aligned on the
    printed checker lines (not the paper edges), to build one large board.

Usage:
  python make_chessboard_board.py
  python make_chessboard_board.py --paper a3 --orientation landscape --grid 3x3
  python make_chessboard_board.py --paper a4

After the physical board exists, place it flat on the visible floor,
photograph it (camera unchanged), and use the follow-up corner-detection
script (next step) to turn each placement into precise Homography anchors.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "chessboard_print"

# (width_cm, height_cm) in portrait orientation
PAPER_SIZES_CM = {
    "a4": (21.0, 29.7),
    "a3": (29.7, 42.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tiled printable chessboard for floor calibration")
    p.add_argument("--squares-x", type=int, default=5, help="number of squares across (columns)")
    p.add_argument("--squares-y", type=int, default=4, help="number of squares down (rows)")
    p.add_argument("--square-cm", type=float, default=20.0, help="physical square edge length (cm)")
    p.add_argument("--quiet-margin-cm", type=float, default=2.0, help="white margin around the pattern (cm)")
    p.add_argument("--paper", choices=["a4", "a3"], default="a3", help="paper size to tile onto")
    p.add_argument(
        "--orientation",
        choices=["auto", "portrait", "landscape"],
        default="auto",
        help="page orientation; auto picks whichever needs fewer pages",
    )
    p.add_argument(
        "--grid",
        default="",
        help='force tiling grid as ROWxCOL, e.g. "3x3" for 9 pages; empty = auto',
    )
    p.add_argument("--page-margin-cm", type=float, default=0.5, help="unprintable margin per page edge (cm)")
    p.add_argument(
        "--overlap-cm",
        type=float,
        default=0.3,
        help="small overlap between adjacent pages for taping (cm); even tiling keeps this exact",
    )
    p.add_argument("--dpi", type=int, default=300, help="print resolution")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    return p.parse_args()


def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size_px)
            except OSError:
                continue
    return ImageFont.load_default()


def build_board_image(
    squares_x: int,
    squares_y: int,
    square_px: int,
    quiet_margin_px: int,
) -> Image.Image:
    pattern_w = squares_x * square_px
    pattern_h = squares_y * square_px
    board_w = pattern_w + 2 * quiet_margin_px
    board_h = pattern_h + 2 * quiet_margin_px

    board = Image.new("L", (board_w, board_h), color=255)
    draw = ImageDraw.Draw(board)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                continue  # keep white; draw only black squares
            x0 = quiet_margin_px + col * square_px
            y0 = quiet_margin_px + row * square_px
            draw.rectangle([x0, y0, x0 + square_px, y0 + square_px], fill=0)
    return board


def make_assembly_guide(
    board: Image.Image,
    tiles: list[dict],
    max_width: int = 1000,
) -> Image.Image:
    scale = min(1.0, max_width / board.width)
    preview = board.convert("RGB").resize(
        (max(1, int(board.width * scale)), max(1, int(board.height * scale))),
        Image.LANCZOS,
    )
    draw = ImageDraw.Draw(preview)
    font = _font(max(14, int(20 * scale * 2)))

    # Draw intended overlap strips first (semi-transparent yellow), then thin
    # borders. Old "full crop rectangles" made a 0.3cm overlap look huge.
    for i, t in enumerate(tiles):
        x0, y0, x1, y1 = t["crop_px"]
        for other in tiles[i + 1 :]:
            ox0, oy0, ox1, oy1 = other["crop_px"]
            ix0, iy0 = max(x0, ox0), max(y0, oy0)
            ix1, iy1 = min(x1, ox1), min(y1, oy1)
            if ix1 > ix0 and iy1 > iy0:
                sx0, sy0, sx1, sy1 = [int(v * scale) for v in (ix0, iy0, ix1, iy1)]
                # Expand hairline overlaps so they stay visible on the preview.
                if sx1 - sx0 < 3:
                    sx1 = sx0 + 3
                if sy1 - sy0 < 3:
                    sy1 = sy0 + 3
                draw.rectangle([sx0, sy0, sx1 - 1, sy1 - 1], fill=(255, 220, 0))

    for t in tiles:
        x0, y0, x1, y1 = [int(v * scale) for v in t["crop_px"]]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=1)
        draw.text((x0 + 6, y0 + 6), t["label"], fill=(255, 0, 0), font=font)
    return preview


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if out_dir.exists():
        for stale in out_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    ppc = args.dpi / 2.54  # pixels per cm
    square_px = int(round(args.square_cm * ppc))
    quiet_margin_px = int(round(args.quiet_margin_cm * ppc))

    board = build_board_image(args.squares_x, args.squares_y, square_px, quiet_margin_px)
    board_w_cm = args.squares_x * args.square_cm + 2 * args.quiet_margin_cm
    board_h_cm = args.squares_y * args.square_cm + 2 * args.quiet_margin_cm

    board_path = out_dir / "board_full_preview.png"
    board.save(board_path)

    portrait_w_cm, portrait_h_cm = PAPER_SIZES_CM[args.paper]

    def page_count(page_w_cm: float, page_h_cm: float) -> tuple[int, int]:
        usable_w = page_w_cm - 2 * args.page_margin_cm
        usable_h = page_h_cm - 2 * args.page_margin_cm
        # Even-tile crop size for a candidate grid; find min rows*cols that fits.
        best: tuple[int, int] | None = None
        for rows in range(1, 8):
            for cols in range(1, 8):
                crop_w = (board_w_cm + (cols - 1) * args.overlap_cm) / cols
                crop_h = (board_h_cm + (rows - 1) * args.overlap_cm) / rows
                if crop_w <= usable_w + 1e-6 and crop_h <= usable_h + 1e-6:
                    if best is None or rows * cols < best[0] * best[1]:
                        best = (rows, cols)
        return best if best is not None else (1, 1)

    if args.orientation == "landscape":
        page_w_cm, page_h_cm = portrait_h_cm, portrait_w_cm
    elif args.orientation == "portrait":
        page_w_cm, page_h_cm = portrait_w_cm, portrait_h_cm
    else:
        rows_p, cols_p = page_count(portrait_w_cm, portrait_h_cm)
        rows_l, cols_l = page_count(portrait_h_cm, portrait_w_cm)
        if rows_l * cols_l < rows_p * cols_p:
            page_w_cm, page_h_cm = portrait_h_cm, portrait_w_cm
        else:
            page_w_cm, page_h_cm = portrait_w_cm, portrait_h_cm

    usable_w_cm = page_w_cm - 2 * args.page_margin_cm
    usable_h_cm = page_h_cm - 2 * args.page_margin_cm

    if args.grid:
        try:
            n_rows_s, n_cols_s = args.grid.lower().split("x")
            n_rows, n_cols = int(n_rows_s), int(n_cols_s)
        except ValueError as exc:
            raise SystemExit('--grid 格式須為 ROWxCOL，例如 "3x3"') from exc
        if n_rows < 1 or n_cols < 1:
            raise SystemExit("--grid 的列/欄必須 >= 1")
    else:
        n_rows, n_cols = page_count(page_w_cm, page_h_cm)

    # Even tiling: each crop is only as large as needed, with EXACT overlap_cm.
    # This avoids the old bug where the last row/col was clamped onto a full
    # printable page and overlapped the previous tile by tens of cm.
    crop_w_cm = (board_w_cm + (n_cols - 1) * args.overlap_cm) / n_cols
    crop_h_cm = (board_h_cm + (n_rows - 1) * args.overlap_cm) / n_rows
    if crop_w_cm > usable_w_cm + 1e-6 or crop_h_cm > usable_h_cm + 1e-6:
        raise SystemExit(
            f"無法用 {n_rows}x{n_cols} 塞進 {args.paper.upper()}："
            f"每塊約 {crop_w_cm:.1f}x{crop_h_cm:.1f} cm，"
            f"可印區只有 {usable_w_cm:.1f}x{usable_h_cm:.1f} cm。"
            f"請加大紙張、減少方格／白邊，或改 --grid。"
        )

    crop_w_px = int(round(crop_w_cm * ppc))
    crop_h_px = int(round(crop_h_cm * ppc))
    step_w_px = int(round((crop_w_cm - args.overlap_cm) * ppc))
    step_h_px = int(round((crop_h_cm - args.overlap_cm) * ppc))
    page_w_px = int(round(page_w_cm * ppc))
    page_h_px = int(round(page_h_cm * ppc))

    label_font = _font(max(18, int(0.35 * ppc)))
    pages: list[Image.Image] = []
    tiles: list[dict] = []

    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * step_w_px
            y0 = row * step_h_px
            x1 = x0 + crop_w_px
            y1 = y0 + crop_h_px
            # Absorb pixel-rounding drift on the last row/col so no gap remains.
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

            crop = board.crop((x0, y0, x1, y1)).convert("RGB")
            # Center crop on a full paper page so "print at 100%" keeps cm exact.
            page = Image.new("RGB", (page_w_px, page_h_px), color=(255, 255, 255))
            ox = (page_w_px - crop.width) // 2
            oy = (page_h_px - crop.height) // 2
            page.paste(crop, (ox, oy))

            label = f"R{row + 1}C{col + 1}"
            draw = ImageDraw.Draw(page)
            draw.text((max(8, ox + 8), max(8, oy + 8)), label, fill=(255, 0, 0), font=label_font)
            draw.text(
                (max(8, ox + 8), min(page_h_px - int(0.5 * ppc), oy + crop.height - int(0.45 * ppc))),
                f"square={args.square_cm:g}cm  overlap={args.overlap_cm:g}cm",
                fill=(255, 0, 0),
                font=_font(max(14, int(0.22 * ppc))),
            )

            page_name = f"page_{label}.png"
            page.save(out_dir / page_name)
            pages.append(page)
            tiles.append({"label": label, "crop_px": (x0, y0, x1, y1), "file": page_name})

    pdf_path = out_dir / "chessboard_pages.pdf"
    if pages:
        pages[0].save(
            pdf_path,
            save_all=True,
            append_images=pages[1:],
            resolution=args.dpi,
        )

    guide = make_assembly_guide(board, tiles)
    guide_path = out_dir / "assembly_guide.png"
    guide.save(guide_path)

    inner_corners_x = args.squares_x - 1
    inner_corners_y = args.squares_y - 1

    print(f"棋盤格：{args.squares_x}x{args.squares_y} 方格，每格 {args.square_cm:g}cm")
    print(f"整塊板子尺寸：約 {board_w_cm:g} x {board_h_cm:g} cm（含 {args.quiet_margin_cm:g}cm 白邊）")
    print(f"內角點數：{inner_corners_x} x {inner_corners_y}（findChessboardCorners 用這個 pattern size）")
    orient_desc = "橫向" if page_w_cm > page_h_cm else "縱向"
    print(
        f"拼頁：{args.paper.upper()}（{orient_desc}），{n_rows} 排 x {n_cols} 欄 = "
        f"{n_rows * n_cols} 頁（每塊約 {crop_w_cm:.1f}x{crop_h_cm:.1f} cm，"
        f"搭接僅 {args.overlap_cm:g}cm）"
    )
    print()
    print("輸出檔案：")
    print(f"  {board_path}（完整板子預覽，非列印用）")
    print(f"  {guide_path}（拼貼指南：紅框＝每頁範圍，數字＝頁碼 R列C欄）")
    print(f"  {pdf_path}（用這份 PDF 列印，務必選「實際尺寸／100%／不要縮放」）")
    print(f"  {out_dir}\\page_R*C*.png（同上內容的個別 PNG，備用）")
    print()
    print("組裝：依 R列C欄 順序排好，沿著印出來的棋盤格線（不是紙緣）對齊黑白格貼合，")
    print("重疊搭接邊只是方便貼合，不影響格子大小。")


if __name__ == "__main__":
    main()
