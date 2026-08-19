"""Printable A4 chessboard for lens calibration (calibrate_lens.py).

Default matches calibrate_lens.py: 7x6 inner corners, 2.5 cm squares
(8x7 squares). Print at 100% / actual size, not "fit to page".

  python make_lens_chessboard.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_OUT = Path(__file__).resolve().parent / "calibration" / "lens_chessboard"
A4_LANDSCAPE_CM = (29.7, 21.0)
DPI = 300


def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size_px)
            except OSError:
                continue
    return ImageFont.load_default()


def cm_to_px(cm: float, dpi: int = DPI) -> int:
    return int(round(cm / 2.54 * dpi))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an A4 lens-calibration chessboard")
    p.add_argument("--cols", type=int, default=7, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square-cm", type=float, default=2.5)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    squares_x = args.cols + 1
    squares_y = args.rows + 1
    square_px = cm_to_px(args.square_cm)
    page_w = cm_to_px(A4_LANDSCAPE_CM[0])
    page_h = cm_to_px(A4_LANDSCAPE_CM[1])
    pattern_w = squares_x * square_px
    pattern_h = squares_y * square_px
    if pattern_w + 40 > page_w or pattern_h + 80 > page_h:
        raise SystemExit("圖案超出 A4 橫式，請縮小 --square-cm 或格數。")

    page = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    x0 = (page_w - pattern_w) // 2
    y0 = (page_h - pattern_h) // 2 - 20
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                continue
            xa = x0 + col * square_px
            ya = y0 + row * square_px
            draw.rectangle([xa, ya, xa + square_px, ya + square_px], fill=(0, 0, 0))

    label = (
        f"Lens chessboard  |  inner {args.cols}x{args.rows}  |  "
        f"square {args.square_cm:g} cm  |  print 100% / actual size"
    )
    font = _font(28)
    draw.text((40, page_h - 55), label, fill=(40, 40, 40), font=font)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"a4_{args.cols}x{args.rows}_{args.square_cm:g}cm"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    page.save(png_path, dpi=(DPI, DPI))
    page.save(pdf_path, "PDF", resolution=DPI, dpi=(DPI, DPI))
    print(f"PNG：{png_path}")
    print(f"PDF：{pdf_path}")
    print("列印選「實際大小 / 100%」，不要「符合頁面」。印完用尺量一格應為 2.5 cm。")


if __name__ == "__main__":
    main()
