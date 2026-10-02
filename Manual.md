# Manual

Daily operator steps: [English.md](English.md) (Chinese: [README.md](README.md)).  
This document is the full technical write-up: pipeline, Homography versions, Stable-ID, cell hold, file map, done vs open issues.  
Chinese original: [說明書.md](說明書.md).

# person-detection-and-tracking

Person detection, tracking, and floor-grid localization from a monitor feed (Tapo C230 / YOLO26 / Homography).  
Pipeline: `YOLO26.track` + BoT-SORT (short) → Stable-ID + OSNet-AIN (long ID) → foot points projected onto the world-coordinate grid. Camera and grid share the same ID colors; the system does not assume a single person.

<p align="center">
  <img src="picture/架構圖.png" alt="System architecture" width="560" />
</p>

## Environment

Use Python **3.14** (tested on 3.14.6) and the project venv `C:\5Gjump\.venv` (VS Code: `Python: Select Interpreter` → `.venv`).

```powershell
cd C:\5Gjump
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first detection run downloads `yolo26s.pt`. The GUI can switch YOLO to `yolo26m` / `yolo26l` (`--model yolo26m.pt`); `--ref pose` pairs the matching `*-pose.pt`. Daily default remains s. Weights are not listed in `requirements.txt` (see comments there).

## detect_grid launcher GUI

Graphical launcher: pick local video or RTSP, YOLO size (s/m/l, default s) and Re-ID, tweak parameters with sliders or spinboxes, then Run → left grid, right camera.

```powershell
python launch_detect_grid.py
```

<p align="center">
  <img src="test/demo_gui_launcher.jpg" alt="detect_grid launcher GUI (test4, 5 Stable-IDs)" width="100%" />
</p>

## Suggested commands

Local `test4` (pose + OSNet-AIN + four-point compensation + A/B/C/O):

```powershell
python detect_grid.py --source test/test4.mp4 --ref pose --cell-hold 2 --quiet --reid-model osnet_ain --error-comp calibration/homography_error_report.json
```

For RTSP, set `--source` to `rtsp://user:pass@IP:554/stream1` (TCP automatic, latest frame only). Single-person video: `test/test.mp4` with `--ref auto`.

Press `q` to quit, `s` to save a frame. Preview max width 1280; windows are resizable.

| Flag | Effect |
|------|--------|
| `--no-pose-skeleton` | Do not draw COCO-17 skeleton |
| `--review-dump` | Write review crops under `test/reid_review/<run-time>/` (not used for live matching; `f00160` = frame 160) |
| `--save-video path.mp4 --no-show --no-realtime` | Record side-by-side demo video |
| `--no-track` | Boxes only, no IDs |
| `--no-floor-grid` | Hide floor marks A/B/C/O |
| `--realtime` / `--no-realtime` | Cap local-video playback to source FPS (never drops tracking frames to catch up) |

<p align="center">
  <img src="test/review_videos/test4_review.webp" width="100%" alt="Demo: camera left, grid right"/>
</p>

Current main demo: `test/review_videos/test4_review.mp4` (`test4` + pose + OSNet-AIN, `--min-hits 16`; includes dirty-crop no-issue, ghost-box hiding, short miss anti-steal, far walker not sticky-stealing IDs). Older localization-only demos still in the repo: `test/demo_v2_chessboard.webp`, `test/demo_v1_manual.webp`.

### Current settings

| Item | Setting |
|------|---------|
| Homography | **v2** (`calibration/homography.json`, **no undistort**) |
| Localization compensation | `--error-comp calibration/homography_error_report.json` |
| Lens intrinsics | `camera_intrinsics.json` exists; **`detect_grid.py` does not apply it** |
| Floor marks | A/B/C/O in `calibration/floor_marks.json` |
| Detection | `yolo26s.pt`, `--ref pose`, `--conf 0.35`, `--cell-hold 2` (each ID sticks to one cell; see below) |
| Short track | BoT-SORT (`trackers/botsort.yaml`; GMC off, short ReID off) |
| Long ID | Stable-ID + `--reid-model osnet_ain`; `--min-hits 16`; `--appear-thresh 0.34` |
| Performance | `--stride 5` (~4 YOLO passes/sec); fixed sampling for files, latest frame for RTSP |
| Review gallery | Off by default; enable with `--review-dump`. Name `fNNNNN` = frame index this run; RTSP ~20 fps (`f00160` ≈ 8 s after start) |

Legacy calib: `--calib calibration/homography_v1_manual.json`. Old desk gray mask: `--valid-xmin 170`.

## Foot point (`--ref pose`)

Detection/tracking still use `yolo26s.pt`; pose runs `yolo26s-pose.pt` separately. Sitting needs 3 consecutive sit evidences. Standing foot fill needs at least 2 full standing poses on the same raw track before using that person’s body-ratio history. With no history, always fall back to box bottom.

| Case | Foot point |
|------|------------|
| Sitting (consecutive sit evidence) | Hip X + box bottom |
| Standing, ankles visible | Ankles |
| Standing, lower body occluded | Fill from same-ID standing history if available; else box bottom |

On-screen legend: green = seat, cyan = ankle, orange = standing fill, red = box bottom, purple = estimate. Skeleton on by default (`kpt-draw-conf ≥ 0.25`); foot colors are separate from skeleton colors. Two models are slower; `--stride 5` still approaches realtime on CPU.

Default `--ref auto`: mid-bottom of the bbox; if the person is clipped by the frame edge, push down from the head.

## Person ID (Stable-ID)

Short track in `trackers/botsort.yaml`: static camera `gmc_method: none`; `with_reid: False` (appearance belongs to Stable-ID—do not put OSNet `.pth` in the yaml); `new_track_thresh: 0.65` reduces door-edge fragment tracks.

Long ID: YOLO boxes people → match gallery → hit keeps ID / continuous track with outfit change stores a new prototype / no hit issues a new ID.

- Live gallery `test/reid_gallery/`: cleared every run; matching uses in-memory vectors
- Review gallery `test/reid_review/`: off by default; `--review-dump` writes crops (see below)
- Gallery only accepts clean, low-overlap boxes
- Re-entry: if OSNet still looks like the same person (including jacket change) → keep ID. After leaving via the **image edge**, if appearance is low and clothes differ a lot → do not reclaim just because “one empty slot remains”; wait `--min-hits` then issue a new ID. Behind-desk misses still reconnect without a new ID
- Stranger tracks are quarantined before issue; overlapping dual boxes keep the older ID; indoor miss ~1.2 s; edge contact clears residual boxes immediately
- `--min-hits 16`, `--stride 5`, ~20 fps → first new ID ~4 s after start (already enrolled people need not wait again). ID set changes print `[ID-CHANGE]`
- Issued appearance is a frozen anchor (not drifted by EMA). If current look unlike the anchor **and** world position jumps too far → restore prototypes and temporarily cancel that ID (HMP-style long memory + ID-switch recovery)

### Review gallery filenames

Path: `test/reid_review/<run-time>/ID001/f00160.jpg`. For humans only; not written back into live vectors.

`f00160` is **not** 1 min 60 s; it is **frame 160 of this run**, counting from the first frame after start.

```
seconds ≈ frame_index ÷ fps
```

| Source | fps | `f00160` | `f00200` | `f01200` |
|------|-----|----------|----------|----------|
| RTSP (Stable-ID fixed at 20 fps) | 20 | 8.0 s | 10.0 s | 1 min 0 s |
| Local video | file-reported FPS | 160 ÷ that fps | 200 ÷ that fps | 1200 ÷ that fps |

Default `--review-every 10`: each issued ID saves at frames 10, 20, 30… (~0.5 s at 20 fps). A missed slot leaves a gap; later names stay on the cadence (not 211, 221). Unlabeled `person` boxes without a stable ID are not saved.

## Stride, realtime, and cell hold

| Mechanism | Purpose | Behavior |
|------|------|------|
| `--stride N` (default 5) | Cut compute | Run YOLO every N frames; in-between frames reuse and coast. Grid label `cached` means reuse |
| `LatestFrameCapture` | Reduce RTSP lag | Drop buffered old frames when inference is slow; always process the newest. **Not** used for local `.mp4` |
| `--cell-hold N` (default 2) | Light cells when someone has an ID; no flicker on grid lines | Cells stick to **people**, not whole-cell voting. See next section |

Local video always processes frames `1, 1+stride, 1+2×stride, …` so the same file yields matching IDs across runs. `--realtime` (on by default) only caps playback to source FPS; if inference is slow, playback slows—it does not drop tracking frames to catch the timeline.

### Why cells used to flicker (`--cell-hold`)

Left boxes/IDs draw whenever someone is present. The right grid used to light a cell only after **N consecutive detections** in that cell, and clear only after N consecutive misses; drawing also required **current foot ∩ confirmed cell**.

When a person stands on a grid line, the foot point jumps between neighboring cells A and B:

| | Old (whole-cell vote) | Now (each ID sticks to one cell) |
|--|-------------------|------------------------|
| First cell estimate | Needs 2 hits to light → ID may exist while cell is still dark | Lights immediately |
| Foot oscillating A↔B | This tick foot in B but confirmed is A → neither matches → **cell goes dark** | Keep previous cell |
| `--cell-hold 1` | Cell follows foot every tick → left/right flicker | Stickiness off; same flicker |

Keep **`--cell-hold 2`** (GUI default is also 2). Do not set 1 just to “light faster.”

True move: switch only after 2 consecutive YOLO hits in the neighbor. Jump across more than one cell (not adjacent) switches immediately. One missed tick keeps the light; two consecutive empty cells clear it. Feet outside the map (`cell` empty) still go dark—that is localization off-map, not line flicker.

Keep the suggested `test4` command:

```powershell
python detect_grid.py --source test/test4.mp4 --ref pose --cell-hold 2 --quiet --reid-model osnet_ain --error-comp calibration/homography_error_report.json
```

## Floor marks (A/B/C/O)

Camera and right-hand grid draw the same floor points (person boxes show ID only). Far-right corner is omitted so projections do not land on desks/furniture. Coordinates live in `calibration/floor_marks.json`.

| Mark | World (cm) | Location |
|----|----------------|------|
| **A** | (170, 450) | Near left |
| **B** | (170, 180) | Far left |
| **C** | (440, 450) | Near right |
| **O** | (260, 315) | Aisle center |

When editing, click only visible floor:

```powershell
python pick_floor_marks.py --source test/static_frame.jpg
python pick_floor_marks.py --source test/test4.mp4 --frame 1
```

Click A → B → C → O in order, press `s` to overwrite `floor_marks.json`.

## Calibration

Three floor Homography versions are kept and never overwrite each other. Default scripts read `calibration/homography.json` (currently **v2, marked on the raw image, no lens undistort**). The launcher “校正” control can switch to **v3 ChArUco** without overwriting that default file.

| Version | File | Method | Notes |
|------|------|------|------|
| **v1** | `calibration/homography_v1_manual.json` | Manual tile corners (`calibrate_boundary.py`) | Click-check error ~**8.9 cm** |
| **v2** | `calibration/homography_v2_chessboard.json` | Large floor chessboard auto corners | Current default; click-check ~**3.8 cm** |
| **v3** | `calibration/homography_charuco.json` | Floor ChArUco intersections (`calibrate_charuco_floor.py`) | Trial only; does not replace v2; ~**2 cm** near the board. Same planar Homography class as chessboard when fully in view |

### Four-point measured compensation (used daily)

Homography is accurate near the board and shows systematic drift farther away (especially near-right). Use `verify_homography.py --measure-error` on 4 known floor points → `calibration/homography_error_report.json`, then `--error-comp` at runtime: Homography first, then a world-space affine.

On those 4 points, average error goes about **32 cm → 6 cm** (max about **90 → 10 cm**). One cell is ~45 cm, so the aisle often stays the same cell; right/extrapolated areas show the gain more. Compensation fixes Homography error; it **cannot** fix foot jitter from behind-desk occlusion. `verify_homography.py` can take the same `--error-comp` so clicks show compensated coordinates.

v3 ChArUco uses a separate `calibration/charuco_floor/error_report.json` (5 points); do not apply the v2 report onto v3.

### Lens intrinsics (on disk, not used daily)

Tapo C230 intrinsics were estimated with a handheld A4 chessboard. **`detect_grid.py` currently neither reads nor undistorts.**

| File | Role |
|------|------|
| `calibration/camera_intrinsics.json` | `cv2.calibrateCamera` intrinsics + distortion |
| `calibration/lens_frames/`, `calibration/lens_chessboard/` | Capture frames / board images |
| `calibrate_lens.py`, `make_lens_chessboard.py` | Capture and estimate |

Tried undistort-then-reestimate floor H: slightly better near the board, but wide-angle edges over-correct (point A can fall outside the image), so it never entered live localization. Re-running `calibrate_chessboard_floor.py calibrate` **auto-undistorts** if `camera_intrinsics.json` is present—do not do that for daily use.

### Coordinate frame and remap

- Virtual top-left is `(0,0)` (OK if not clickable)
- Floor grid about `X 0–530 cm`, `Y 0–540 cm`
- Tiles: first left cell 35 cm, rest 45 cm
- Use `make_floor_grid.py` to enter size and cell counts; split saved in `calibration/floor_grid.json` (builtin this-room edges if the file is missing)
- Left side is no longer drawn as a light desk zone by default (`--valid-xmin 0`); restore old mask with `--valid-xmin 170`

```powershell
# Measure error / refresh four-point report
python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg

# v2 chessboard (capture then calibrate; do not let it auto-apply intrinsics)
python calibrate_chessboard_floor.py capture --source "rtsp://user:pass@camera-IP:554/stream1"
python calibrate_chessboard_floor.py calibrate --image calibration/chessboard_floor/capture.jpg --origin-x 190 --origin-y 400 --out calibration/homography_v2_chessboard.json

# v3 ChArUco (writes homography_charuco.json; does not overwrite v2)
python calibrate_charuco_floor.py --capture
python verify_homography.py --measure-error --calib calibration/homography_charuco.json --image calibration/charuco_floor/capture.jpg --out calibration/charuco_floor/error_report.json

# v1 manual clicks
python calibrate_boundary.py --width 530 --height 540
python verify_homography.py
```

## Status

**Done**

- YOLO26 `model.track` + BoT-SORT; long ID: Stable-ID + OSNet-AIN
- `--ref pose`: sit = hip + box bottom; stand + occluded = same-ID standing history fill, else box bottom
- Re-entry: keep ID if still looks like the same person after outfit change; new ID only after leaving via the image edge with clearly unlike appearance/clothes
- Overlapping dual-box merge, harder new tracks from door fragments, short indoor miss keep / edge clear
- Local video fixed tracking frames; RTSP latest frame + TCP; cell hold / stride; A/B/C/O
- Four-point measured compensation; lens intrinsics on disk but no daily undistort

**Open**

- Crowded occlusion / misses can still flicker IDs; near-camera large boxes may swallow another person
- `test3.mp4` crosses often; `--min-hits 16` is conservative
- Look-alike body/clothes swaps can still reclaim an old ID via OSNet; wide-angle edges are not fixed by intrinsics alone
- Behind-desk occlusion can still make the foot jump cells

## Other scripts

```powershell
python launch_detect_grid.py
python test_rtsp.py "rtsp://user:pass@camera-IP:554/stream1"
python grid_occupancy.py
python grid_occupancy.py --x 215 --y 360
python make_floor_grid.py --width 530 --height 540 --cols 12 --rows 12 --x-first 35 --no-show
```

- `launch_detect_grid.py`: see “detect_grid launcher GUI” above.
- `test_rtsp.py`: stream-only test (TCP, preview ≤1280, still reads 2880×1620). Headless: `--no-preview --frames 60`
- `grid_occupancy.py`: click camera floor to see which cell lights (yellow = occupied). Scale reference: `test/floor_grid_generated.jpg`
- `make_floor_grid.py`: GUI for size and cell counts; writes the grid image and `calibration/floor_grid.json`

## Reports

- [9/18](PPT%20report/報告9_18.pdf) ([PPT](PPT%20report/報告9_18.pptx))
- [8/21](PPT%20report/報告8_21.pdf) ([PPT](PPT%20report/報告8_21.pptx))
- [8/7](PPT%20report/報告8_7.pdf) ([PPT](PPT%20report/報告8_7.pptx))
- [7/24](PPT%20report/報告7_24.pdf) ([PPT](PPT%20report/報告7_24.pptx))
- [7/10](PPT%20report/報告7_10.pdf) ([PPT](PPT%20report/報告7_10.pptx))
