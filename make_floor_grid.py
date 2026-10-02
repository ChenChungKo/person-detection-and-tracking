"""Plan a world-coordinate floor grid and export a labelled diagram.

GUI (default):
  python make_floor_grid.py

CLI:
  python make_floor_grid.py --width 530 --height 540 --cols 12 --rows 12 --x-first 35 --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from grid_occupancy import (
    FLOOR_GRID_IMAGE,
    FLOOR_GRID_JSON,
    apply_floor_grid,
    axis_edges,
    draw_grid,
    imwrite_unicode,
    save_floor_grid,
)


def _sizes_from_edges(edges: list[float]) -> list[float]:
    return [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]


def _fmt_sizes(sizes: list[float]) -> str:
    return ", ".join(f"{s:g}" for s in sizes)


def _opt_float(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    return float(text)


def build_grid(
    width: float,
    height: float,
    cols: int,
    rows: int,
    x_first: float | None,
    x_last: float | None,
    y_first: float | None,
    y_last: float | None,
) -> tuple[list[float], list[float], np.ndarray]:
    x_edges = axis_edges(width, cols, x_first, x_last)
    y_edges = axis_edges(height, rows, y_first, y_last)
    apply_floor_grid(x_edges, y_edges)
    grid = draw_grid(None, landmarks=False)
    return x_edges, y_edges, grid


def save_grid(
    x_edges: list[float],
    y_edges: list[float],
    grid: np.ndarray,
    json_path: Path,
    img_path: Path,
    extra: dict,
) -> None:
    save_floor_grid(json_path, x_edges, y_edges, extra=extra)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(img_path, grid)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a floor-grid diagram from room size")
    p.add_argument("--gui", action="store_true", help="開 GUI（沒給長寬時預設就開）")
    p.add_argument("--width", type=float, default=None, help="X 總寬 cm")
    p.add_argument("--height", type=float, default=None, help="Y 總長 cm")
    p.add_argument("--cols", type=int, default=None, help="X 切幾格")
    p.add_argument("--rows", type=int, default=None, help="Y 切幾格")
    p.add_argument("--x-first", type=float, default=None, help="X 頭格 cm；省略則 X 全等分")
    p.add_argument("--x-last", type=float, default=None, help="X 尾格 cm；省略則尾格與中間相同")
    p.add_argument("--y-first", type=float, default=None, help="Y 頭格 cm；省略則 Y 全等分")
    p.add_argument("--y-last", type=float, default=None, help="Y 尾格 cm；省略則尾格與中間相同")
    p.add_argument("--json", default=str(FLOOR_GRID_JSON), help="切法輸出")
    p.add_argument("--out", default=str(FLOOR_GRID_IMAGE), help="格子圖輸出")
    p.add_argument("--no-show", action="store_true", help="CLI 模式不開預覽窗")
    return p.parse_args()


class FloorGridGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("產生地板格子圖")
        self.minsize(980, 640)
        self.geometry("1100x720")

        self.width_cm = tk.StringVar(value="530")
        self.height_cm = tk.StringVar(value="540")
        self.cols = tk.StringVar(value="12")
        self.rows = tk.StringVar(value="12")
        self.x_ends = tk.BooleanVar(value=True)
        self.y_ends = tk.BooleanVar(value=False)
        self.x_first = tk.StringVar(value="35")
        self.x_last = tk.StringVar(value="")
        self.y_first = tk.StringVar(value="")
        self.y_last = tk.StringVar(value="")
        self.status = tk.StringVar(value="填長寬與格數，按預覽。頭尾可較小，中間等分。")
        self._bgr: np.ndarray | None = None
        self._x_edges: list[float] | None = None
        self._y_edges: list[float] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._load_existing()
        self._build()
        self.after(200, self._preview)

    def _load_existing(self) -> None:
        path = FLOOR_GRID_JSON
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return
        if "width_cm" in data:
            self.width_cm.set(str(data["width_cm"]))
        if "height_cm" in data:
            self.height_cm.set(str(data["height_cm"]))
        if "n_cols" in data:
            self.cols.set(str(data["n_cols"]))
        if "n_rows" in data:
            self.rows.set(str(data["n_rows"]))
        xf, xl = data.get("x_first_cm"), data.get("x_last_cm")
        yf, yl = data.get("y_first_cm"), data.get("y_last_cm")
        self.x_ends.set(xf is not None or xl is not None)
        self.y_ends.set(yf is not None or yl is not None)
        self.x_first.set("" if xf is None else str(xf))
        self.x_last.set("" if xl is None else str(xl))
        self.y_first.set("" if yf is None else str(yf))
        self.y_last.set("" if yl is None else str(yl))

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(1, weight=1)

        ttk.Label(frm, text="規劃地板格子", font=("", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        left = ttk.Frame(frm)
        left.grid(row=1, column=0, sticky="nsw", padx=(0, 12))
        preview = ttk.Frame(frm)
        preview.grid(row=1, column=1, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)

        x_box = ttk.LabelFrame(left, text="X", padding=8)
        x_box.pack(fill=tk.X, pady=(0, 8))
        self._row(x_box, 0, "總寬 cm", self.width_cm)
        self._row(x_box, 1, "切幾格", self.cols)
        ttk.Checkbutton(
            x_box,
            text="頭尾較小，中間等分",
            variable=self.x_ends,
            command=self._sync_ends,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._x_first_ent = self._row(x_box, 3, "頭格 cm", self.x_first)
        self._x_last_ent = self._row(x_box, 4, "尾格 cm（空白＝與中間相同）", self.x_last, width=10)

        y_box = ttk.LabelFrame(left, text="Y", padding=8)
        y_box.pack(fill=tk.X, pady=(0, 8))
        self._row(y_box, 0, "總長 cm", self.height_cm)
        self._row(y_box, 1, "切幾格", self.rows)
        ttk.Checkbutton(
            y_box,
            text="頭尾較小，中間等分",
            variable=self.y_ends,
            command=self._sync_ends,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._y_first_ent = self._row(y_box, 3, "頭格 cm", self.y_first)
        self._y_last_ent = self._row(y_box, 4, "尾格 cm（空白＝與中間相同）", self.y_last, width=10)

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=8)
        ttk.Button(btns, text="預覽", command=self._preview).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="儲存格子圖", command=self._save).pack(side=tk.LEFT)

        ttk.Label(left, textvariable=self.status, wraplength=320, foreground="#444").pack(
            fill=tk.X, pady=(8, 0)
        )

        self.canvas = tk.Canvas(preview, background="#f5f5f5", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._paint())
        self._sync_ends()

    def _row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        var: tk.StringVar,
        width: int = 10,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ent = ttk.Entry(parent, textvariable=var, width=width)
        ent.grid(row=row, column=1, sticky="e", pady=2, padx=(8, 0))
        return ent

    def _sync_ends(self) -> None:
        x_state = tk.NORMAL if self.x_ends.get() else tk.DISABLED
        y_state = tk.NORMAL if self.y_ends.get() else tk.DISABLED
        self._x_first_ent.configure(state=x_state)
        self._x_last_ent.configure(state=x_state)
        self._y_first_ent.configure(state=y_state)
        self._y_last_ent.configure(state=y_state)

    def _values(self) -> tuple[float, float, int, int, float | None, float | None, float | None, float | None]:
        width = float(self.width_cm.get())
        height = float(self.height_cm.get())
        cols = int(round(float(self.cols.get())))
        rows = int(round(float(self.rows.get())))
        x_first = x_last = y_first = y_last = None
        if self.x_ends.get():
            x_first = _opt_float(self.x_first.get())
            x_last = _opt_float(self.x_last.get())
            if x_first is None and x_last is None:
                raise ValueError("X 勾了頭尾較小，請填頭格或尾格")
        if self.y_ends.get():
            y_first = _opt_float(self.y_first.get())
            y_last = _opt_float(self.y_last.get())
            if y_first is None and y_last is None:
                raise ValueError("Y 勾了頭尾較小，請填頭格或尾格")
        return width, height, cols, rows, x_first, x_last, y_first, y_last

    def _preview(self) -> None:
        try:
            width, height, cols, rows, xf, xl, yf, yl = self._values()
            x_edges, y_edges, grid = build_grid(width, height, cols, rows, xf, xl, yf, yl)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("格子", str(exc))
            return
        self._x_edges = x_edges
        self._y_edges = y_edges
        self._bgr = grid
        self.status.set(
            f"X：{_fmt_sizes(_sizes_from_edges(x_edges))} cm（總寬 {width:g}）\n"
            f"Y：{_fmt_sizes(_sizes_from_edges(y_edges))} cm（總長 {height:g}）"
        )
        self._paint()

    def _save(self) -> None:
        if self._bgr is None or self._x_edges is None or self._y_edges is None:
            self._preview()
        if self._bgr is None or self._x_edges is None or self._y_edges is None:
            return
        try:
            width, height, cols, rows, xf, xl, yf, yl = self._values()
            extra = {
                "x_first_cm": xf,
                "x_last_cm": xl,
                "y_first_cm": yf,
                "y_last_cm": yl,
            }
            save_grid(
                self._x_edges,
                self._y_edges,
                self._bgr,
                FLOOR_GRID_JSON,
                FLOOR_GRID_IMAGE,
                extra,
            )
        except (ValueError, tk.TclError, OSError) as exc:
            messagebox.showerror("儲存", str(exc))
            return
        self.status.set(
            f"已存切法 {FLOOR_GRID_JSON.name} 與格子圖 {FLOOR_GRID_IMAGE.name}"
        )
        messagebox.showinfo(
            "已儲存",
            f"切法：{FLOOR_GRID_JSON}\n格子圖：{FLOOR_GRID_IMAGE}",
        )

    def _paint(self) -> None:
        if self._bgr is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        h, w = self._bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(self._bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo = photo
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=photo)


def run_cli(args: argparse.Namespace) -> None:
    if args.width is None or args.height is None or args.cols is None or args.rows is None:
        raise SystemExit("CLI 請給 --width --height --cols --rows，或不要帶參數改開 GUI。")
    x_edges, y_edges, grid = build_grid(
        args.width,
        args.height,
        args.cols,
        args.rows,
        args.x_first,
        args.x_last,
        args.y_first,
        args.y_last,
    )
    extra = {
        "x_first_cm": args.x_first,
        "x_last_cm": args.x_last,
        "y_first_cm": args.y_first,
        "y_last_cm": args.y_last,
    }
    json_path = Path(args.json)
    img_path = Path(args.out)
    save_grid(x_edges, y_edges, grid, json_path, img_path, extra)
    print(f"X 邊：{_fmt_sizes(_sizes_from_edges(x_edges))} cm  →  總寬 {args.width:g}")
    print(f"Y 邊：{_fmt_sizes(_sizes_from_edges(y_edges))} cm  →  總長 {args.height:g}")
    print(f"切法：{json_path}")
    print(f"格子圖：{img_path}")
    if args.no_show:
        return
    cv2.namedWindow("Floor grid", cv2.WINDOW_NORMAL)
    cv2.imshow("Floor grid", grid)
    print("預覽：按 q 關閉")
    while True:
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    use_gui = args.gui or all(
        getattr(args, name) is None for name in ("width", "height", "cols", "rows")
    )
    if use_gui:
        FloorGridGui().mainloop()
        return
    run_cli(args)


if __name__ == "__main__":
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
