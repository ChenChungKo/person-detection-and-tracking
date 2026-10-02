# person-detection-and-tracking

Tapo C230 camera: detect people, assign stable IDs, and project foot points onto a planned floor grid. For principles and limits, see [Manual.md](Manual.md) (Chinese: [說明書.md](說明書.md)). The Chinese step guide is [README.md](README.md).

![System architecture](picture/架構圖.png)

Daily defaults: chessboard Homography v2, four-point compensation, YOLO26s + pose, OSNet-AIN.  
Do not undistort in the live pipeline. Do not overwrite `calibration/homography.json` directly with ChArUco or manual tile clicks. Do not write OSNet into `trackers/botsort.yaml`.

## Demo video

Left = camera, right = floor grid. Main clip: [test4_review.mp4](test/review_videos/test4_review.mp4)

![Demo: camera left, grid right](test/review_videos/test4_review.webp)

Older comparisons: [chessboard v2](test/demo_v2_chessboard.mp4), [manual tile clicks v1](test/demo_v1_manual.mp4).

---

## 1. Set up the environment

Use **Python 3.14** (tested on 3.14.6) for the project virtual environment. In VS Code, select Interpreter → `.venv`.

```powershell
cd C:\5Gjump
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You can then run the scripts below. The first detection run downloads `yolo26s.pt` and `yolo26s-pose.pt` (weights are not listed in `requirements.txt`; see the comments there).

---

## 2. Plan the floor grid

The grid is **measured and split in world coordinates first**, not cropped from the camera image. For a new site or camera, measure that room’s floor again. Virtual top-left is `(0, 0)` (OK if blocked and not clickable); Y increases toward the camera.

Open the GUI, enter total length, total width, and cell counts. Optionally enable “smaller head/tail cells, equal middle cells” (頭尾較小，中間等分). This room’s defaults: X 530 cm, 12 cells, first cell 35 cm; Y 540 cm, 12 equal cells. Preview, then save.

```powershell
python make_floor_grid.py
```

![Floor-grid planner GUI](test/demo_floor_grid_gui.jpg)

You get `test/floor_grid_generated.jpg` and the split in `calibration/floor_grid.json` (detection and grid views use this file).

---

## 3. Lay the board and calibrate Homography (three methods)

Lay the board flat on the floor and fully in view. The origin is numbered inner corner **#0** (about 190, 400 in this room); the near-left outer paper corner is about 170, 420.

Write a new file first; only after you confirm, copy it to the live `calibration/homography.json`. The three methods do not overwrite each other. **Daily method: chessboard.** Do not apply lens undistortion.

### 3.1 Chessboard (daily)

Corners are detected automatically. In the detection GUI, pick calibration 「棋盤」 (chessboard). The image defaults to the capture just taken: `calibration/chessboard_floor/capture.jpg`.

Capture one frame:

```powershell
python calibrate_chessboard_floor.py capture --source "rtsp://user:pass@IP:554/stream1"
```

Compute with origin (190, 400) and save a new file first:

```powershell
python calibrate_chessboard_floor.py calibrate --origin-x 190 --origin-y 400 --out calibration/homography_v2_chessboard.json
```

When origin #0 looks correct on the corner image, overwrite the Homography that detection actually reads:

```powershell
Copy-Item calibration\homography_v2_chessboard.json calibration\homography.json
```

![Chessboard corners and origin #0](calibration/chessboard_floor/origin_marked.jpg)

### 3.2 ChArUco (trial)

Intersections are numbered. Writes only `homography_charuco.json`; it does not replace the chessboard file. Detection uses it only when the GUI calibration is 「ChArUco」. Compensation uses `calibration/charuco_floor/error_report.json`, not the chessboard report from step 5.

```powershell
python calibrate_charuco_floor.py --capture
```

Produces `calibration/homography_charuco.json`.

![ChArUco corners and origin #0](calibration/charuco_floor/detected_corners.jpg)

### 3.3 Manual tile corners (legacy)

Click visible tile corners on the image and enter each point’s world coordinates. Larger error; do not use as the daily default.

```powershell
python calibrate_boundary.py --width 530 --height 540 --out calibration/homography_v1_manual.json
```

Produces `homography_v1_manual.json`.

![Manual tile-corner clicks](calibration/v1_tile_clicks.jpg)

---

## 4. Floor reference marks

Only after Homography is ready, click visible floor points and save `calibration/floor_marks.json`. Detection and the grid label these four marks.

| Mark | World (cm) | Location |
| --- | ---------- | ---- |
| A | (170, 450) | Near left |
| B | (170, 180) | Far left |
| C | (440, 450) | Near right |
| O | (260, 315) | Aisle center |

Click A → B → C → O in order, then press `s` to save:

```powershell
python pick_floor_marks.py --source test/static_frame.jpg
```

![Floor marks A B C O](test/floor_overlay_cam.jpg)

Click the camera floor once to see which grid cell lights up:

```powershell
python grid_occupancy.py
```

Skip clicking and light a cell from world coordinates (215, 360) cm:

```powershell
python grid_occupancy.py --x 215 --y 360
```

---

## 5. Verify localization and four-point compensation

Click the floor to see predicted world coordinates (cm):

```powershell
python verify_homography.py
python verify_homography.py --error-comp calibration/homography_error_report.json
```

Accuracy is good near the board and drifts farther away (especially near-right). Measure 4 known floor points and save a compensation file; the GUI 「四點補償」 checkbox and `--error-comp` use it. Do not apply the v2 compensation report to ChArUco.

```powershell
python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg
```

Produces `calibration/homography_error_report.json`.

![Four-point measured compensation](test/error_comp_four_points.jpg)

---

## 6. Run detection

```powershell
python launch_detect_grid.py
```

After Run: left = grid, right = camera. Defaults: local video, chessboard calibration, foot ref `pose`, YOLO `yolo26s`, Re-ID `osnet_ain`, with skeleton, tracking/ID, four-point compensation, and A/B/C/O enabled.

Adjustable options:

- Source: local video or RTSP
- Calibration: chessboard, ChArUco
- Foot ref: pose, auto, foot, head_drop
- Checkboxes: skeleton, tracking/ID, four-point compensation, A/B/C/O, quiet log, review crops
- YOLO: yolo26s, yolo26m, yolo26l
- Re-ID: osnet_ain, osnet_ibn, osnet, none
- Sliders: conf, stride, cell-hold, min-hits, out-margin

![detect_grid running: five people on grid and camera](test/demo_detect_five.jpg)

Multi-person local clip `test/test4.mp4` (or the command below). Press `q` to quit, `s` to save a frame. Demo: `test/review_videos/test4_review.mp4`.

```powershell
python detect_grid.py --source test/test4.mp4 --ref pose --cell-hold 2 --quiet --reid-model osnet_ain --error-comp calibration/homography_error_report.json
```

Live: choose RTSP in the GUI, URL `rtsp://user:pass@IP:554/stream1` (this site: `rtsp://oriongo:123456789@192.168.0.200:554/stream1`). Uses TCP automatically and only the latest frame. Stream-only test:

```powershell
python test_rtsp.py "rtsp://user:pass@camera-IP:554/stream1"
```

For the single-person clip `test/test.mp4`, set foot ref to `auto`.

---

## Folders

- `PPT report`: progress reports; each date has PDF and PPT.
- `calibration`: calibration outputs. Detection reads `homography.json` (chessboard v2), `floor_grid.json`, `floor_marks.json`, and `homography_error_report.json`. Chessboard, ChArUco, and manual tile each keep their own file and do not overwrite each other. `chessboard_print` / `charuco_print` are printable boards; `chessboard_floor` / `charuco_floor` are captured calibration images. `camera_intrinsics.json` and `lens_frames` are lens intrinsics; daily detection does not undistort.
- `picture`: architecture and timeline figures.
- `test`: demo videos, step screenshots, single-person `test.mp4`, multi-person `test4.mp4`, static frame `static_frame.jpg`. Side-by-side review clip: `review_videos/test4_review.mp4`.
- `trackers`: short-track config `botsort.yaml` (Re-ID off). Long-term ID uses OSNet; do not write it into this file.

---

## Docs and reports

- Full English notes: [Manual.md](Manual.md)
- Full Chinese notes: [說明書.md](說明書.md)
- Chinese steps: [README.md](README.md)
- [9/18](PPT%20report/報告9_18.pdf) ([PPT](PPT%20report/報告9_18.pptx))
- [8/21](PPT%20report/報告8_21.pdf) ([PPT](PPT%20report/報告8_21.pptx))
- [8/7](PPT%20report/報告8_7.pdf) ([PPT](PPT%20report/報告8_7.pptx))
- [7/24](PPT%20report/報告7_24.pdf) ([PPT](PPT%20report/報告7_24.pptx))
- [7/10](PPT%20report/報告7_10.pdf) ([PPT](PPT%20report/報告7_10.pptx))
