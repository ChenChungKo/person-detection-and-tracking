# person-detection-and-tracking

監視器畫面做人體偵測、追蹤與地板格子定位（Tapo C230 / YOLO26 / Homography）。  
管線：`YOLO26.track` + BoT-SORT（短期）→ Stable-ID + OSNet-AIN（長期 ID）→ 腳點投到世界座標格子。畫面與格子用同一套 ID 顏色，不假設場上只有一人。

<p align="center">
  <img src="picture/架構圖.png" alt="系統架構圖" width="560" />
</p>

## 環境

請用專案虛擬環境 `C:\5Gjump\.venv`（VS Code：`Python: Select Interpreter` → `.venv`）。

```powershell
cd C:\5Gjump
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

首次偵測會下載 `yolo26s.pt`（也可改 `yolo26n.pt` / `yolo26m.pt`）。`--ref pose` 會再載入 `yolo26s-pose.pt`。

## 建議指令

本機 `test4`（pose + OSNet-AIN + 四點補償 + A/B/C/O）：

```powershell
python detect_grid.py --source test/test4.mp4 --ref pose --cell-hold 2 --quiet --reid-model osnet_ain --error-comp calibration/homography_error_report.json
```

RTSP 把 `--source` 換成 `rtsp://帳號:密碼@IP:554/stream1` 即可（自動 TCP、只處理最新幀）。單人影片可用 `test/test.mp4` 並改 `--ref auto`。

按 `q` 結束，`s` 存圖。預覽寬度上限 1280，視窗可拖曳縮放。

| 加上 | 作用 |
|------|------|
| `--no-pose-skeleton` | 不畫 COCO-17 骨架 |
| `--review-dump` | 裁圖寫入 `test/reid_review/<時間>/`（不參與即時比對） |
| `--save-video 路徑.mp4 --no-show --no-realtime` | 錄左右對照影片 |
| `--no-track` | 只要人框、不要 ID |
| `--no-floor-grid` | 關掉地上 A/B/C/O |
| `--realtime` / `--no-realtime` | 本機影片是否限播放速度（都不丟追蹤幀） |

<p align="center">
  <img src="test/demo_stable_id_osnet_ain.webp" width="100%" alt="Demo：左偵測右格子"/>
</p>

目前主 demo：`test/demo_stable_id_osnet_ain.mp4`（同上設定、`--min-hits 16`）。舊版僅定位對照仍在倉庫：`test/demo_v2_chessboard.webp`、`test/demo_v1_manual.webp`。

### 目前設定

| 項目 | 設定 |
|------|------|
| Homography | **v2**（`calibration/homography.json`，**不去畸變**） |
| 定位補償 | `--error-comp calibration/homography_error_report.json` |
| 鏡頭內參 | `camera_intrinsics.json` 有檔；**`detect_grid.py` 不套用** |
| 定位對照 | A/B/C/O，見 `calibration/floor_marks.json` |
| 偵測 | `yolo26s.pt`、`--ref pose`、`--conf 0.45`、`--cell-hold 2` |
| 短追蹤 | BoT-SORT（`trackers/botsort.yaml`；GMC off、短 ReID off） |
| 長期 ID | Stable-ID + `--reid-model osnet_ain`；`--min-hits 16`；`--appear-thresh 0.34` |
| 效能 | `--stride 5`（約每秒 4 次 YOLO）；本機固定取樣、RTSP 最新幀 |
| 審查庫 | 預設關 |

舊校正：`--calib calibration/homography_v1_manual.json`。舊桌區灰格：`--valid-xmin 170`。舊短追蹤：`--tracker trackers/bytetrack_stable.yaml`。

## 腳點（`--ref pose`）

偵測／追蹤仍用 `yolo26s.pt`，姿態另跑 `yolo26s-pose.pt`。坐姿需連續 3 次證據才當坐；站立補腳需同一 raw track 先累積至少 2 次完整站姿，之後才用該人歷史身體比例補腳。沒有歷史一律退回框底。

| 情況 | 腳點 |
|------|------|
| 坐（連續坐姿證據） | 髖 X + 框底 |
| 站、腳踝可見 | 腳踝 |
| 站、下半身被擋 | 同一 ID 有完整站姿歷史才補腳；否則框底 |

畫面上方圖例：綠＝座位、青＝腳踝、橘＝站立補腳、紅＝框底、紫＝推估。骨架預設開啟（`kpt-draw-conf ≥ 0.25`）；腳點顏色與骨架顏色分開。雙模型較慢，`--stride 5` 在 CPU 上仍可接近即時。

預設 `--ref auto` 時：bbox 底邊中點；人被畫面裁切時改頭頂下推。

## 人物 ID（Stable-ID）

短追蹤在 `trackers/botsort.yaml`：固定監視器 `gmc_method: none`；`with_reid: False`（外貌交給 Stable-ID，不要把 OSNet `.pth` 寫進 yaml）；`new_track_thresh: 0.65` 減少門邊碎框開新軌。

長期 ID：YOLO 框人 → 比對圖庫 → 命中沿用／連續追蹤換裝則存新原型／都沒中才發新號。

- 即時圖庫 `test/reid_gallery/`：每次重跑清空；比對用記憶體向量
- 審查庫 `test/reid_review/`：預設關；`--review-dump` 才寫裁圖
- 圖庫只收乾淨、低重疊框
- 回場：OSNet 仍像同一人（含換外套）→ 沿用。從畫面**邊緣**離開後，外貌低且衣服差很多 → 不因「只剩一個空號」收回，等 `--min-hits` 後發新號。桌後漏檢仍接回、不發新號
- 陌生軌先隔離再發號；雙框合併留舊號；室內漏檢約 1.2 秒；貼邊立刻清除殘框
- `--min-hits 16`、`--stride 5`、約 20 fps → 開頭約 4 秒才出現第一個新號（已登錄的人再出現不必再等）。ID 變化會印 `[ID-CHANGE]`

## 跳幀、即時與格子防抖

| 機制 | 目的 | 行為 |
|------|------|------|
| `--stride N`（預設 5） | 降低運算量 | 每 N 幀跑一次 YOLO；中間幀沿用並預測跟上。格子標 `cached` 代表沿用 |
| `LatestFrameCapture` | RTSP 降低落後感 | 推論慢時丟緩衝區舊幀，永遠處理最新畫面。本機 `.mp4` **不**啟用 |
| `--cell-hold N`（預設 2） | 格子不閃 | 連續 N **次偵測**一致才亮／滅（跟 stride 搭配時只算真正跑 YOLO 的幀） |

本機影片固定處理第 `1, 1+stride, 1+2×stride, …` 幀，同一支影片重跑 ID 才對得上。`--realtime`（預設開）只限制播放不超過來源 FPS，推論慢時會變慢，但不會為了搶時間軸而丟追蹤幀。

## 定位對照（A/B/C/O）

相機與右側格子畫同一組地上點（人框只標 ID）。不含遠右角，避免投到桌子／家具上。座標在 `calibration/floor_marks.json`。

| 點 | 世界座標 (cm) | 位置 |
|----|----------------|------|
| **A** | (170, 450) | 近左 |
| **B** | (170, 180) | 遠左 |
| **C** | (440, 450) | 近右 |
| **O** | (260, 315) | 走道中心 |

改點時只點看得見的地面：

```powershell
python pick_floor_marks.py --source test/static_frame.jpg
python pick_floor_marks.py --source test/test4.mp4 --frame 1
```

依序點 A → B → C → O，按 `s` 覆寫 `floor_marks.json`。

## 校正

地板 Homography 兩個版本都保留、互不覆蓋。預設腳本讀 `calibration/homography.json`（目前＝**v2，原始畫面標定，未套鏡頭去畸變**）。

| 版本 | 檔案 | 作法 | 備註 |
|------|------|------|------|
| **v1** | `calibration/homography_v1_manual.json` | 手動點磁磚角（`calibrate_boundary.py`） | 點擊量測誤差約 **8.9 cm** |
| **v2** | `calibration/homography_v2_chessboard.json` | 地板大棋盤自動角點 | 目前預設；點擊量測約 **3.8 cm** |

### 四點實測補償（日常有在用）

Homography 在棋盤附近準，離板子遠（尤其近端右側）會有系統性偏差。用 `verify_homography.py --measure-error` 在地上量 4 個已知點，存成 `calibration/homography_error_report.json`，定位時加 `--error-comp`：先 Homography，再世界座標 affine。

在這 4 個點上，平均約 **32 cm → 6 cm**（最大約 **90 → 10 cm**）。格子一格約 45 cm，中間走道常常還是同一格，右側／外推比較看得出差。補償修的是 Homography，**不能**修桌後遮擋造成的腳點亂跳。`verify_homography.py` 也可加同一份 `--error-comp`，點擊時看補償後座標。

### 鏡頭內參（有檔、日常不用）

A4 手持棋盤估過 Tapo C230 內參，**`detect_grid.py` 目前不讀、也不去畸變**。

| 檔案 | 用途 |
|------|------|
| `calibration/camera_intrinsics.json` | `cv2.calibrateCamera` 內參＋畸變 |
| `calibration/lens_frames/`、`calibration/lens_chessboard/` | 內參拍攝／棋盤圖 |
| `calibrate_lens.py`、`make_lens_chessboard.py` | 拍幀、估內參 |

試過先 `undistort` 再重估地板 H：棋盤附近略好，廣角邊緣矯正過強（A 會算出畫面外），所以沒接進即時定位。重跑 `calibrate_chessboard_floor.py calibrate` 若偵測到 `camera_intrinsics.json` 會**自動去畸變**——目前不要這樣做。

### 座標系與重標

- 虛擬左上角為 `(0,0)`（點不到也沒關係）
- 地板格約 `X 0~530 cm`、`Y 0~540 cm`
- 地磚：左側第一格 35 cm，其餘 45 cm
- 預設不再把左側畫成淺灰桌區（`--valid-xmin 0`）；恢復舊遮罩：`--valid-xmin 170`

```powershell
# 量測誤差／更新四點報告
python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg

# v2 棋盤（先 capture 再 calibrate；不要讓它自動套內參）
python calibrate_chessboard_floor.py capture --source "rtsp://帳號:密碼@攝影機IP:554/stream1"
python calibrate_chessboard_floor.py calibrate --image calibration/chessboard_floor/capture.jpg --origin-x 190 --origin-y 400 --out calibration/homography_v2_chessboard.json

# v1 手動點選
python calibrate_boundary.py --width 530 --height 540
python verify_homography.py
```

## 狀態

**已完成**

- YOLO26 `model.track` + BoT-SORT；長期 ID：Stable-ID + OSNet-AIN
- `--ref pose`：坐＝髖＋框底；站＋被擋＝同一 ID 的站姿歷史補腳，否則框底
- 回場：換裝仍像同一人則沿用；從畫面邊緣離開且外貌／衣服明顯不像才發新號
- 重疊雙框合併、門邊碎框較難開新號、室內漏檢短沿用／貼邊立刻清
- 本機影片固定追蹤幀；RTSP 最新幀 + TCP；格子防抖／跳幀；A/B/C/O
- 四點實測補償；鏡頭內參有檔但日常不去畸變

**尚未解決**

- 多人遮擋／漏檢仍可能閃號；近鏡頭大框易吃到另一人
- `test3.mp4` 交叉頻繁，`--min-hits 16` 偏保守
- 體型／衣服都很像的換人，OSNet 仍可能配回舊號；廣角邊緣單靠內參去畸變效果不佳
- 桌後遮擋時腳點仍可能跳格

## 其它腳本

```powershell
python test_rtsp.py "rtsp://帳號:密碼@攝影機IP:554/stream1"
python detect_person.py --source test/test.mp4 --no-map
python grid_occupancy.py
python grid_occupancy.py --x 215 --y 360
python export_demo_video.py
```

- `test_rtsp.py`：只測串流（TCP、預覽 ≤1280，仍讀 2880×1620）。無視窗：`--no-preview --frames 60`
- `detect_person.py --no-map`：只要 YOLO 人框
- `grid_occupancy.py`：點監視器地板看格子（黃＝佔用）。刻度參考 `test/floor_grid_generated.jpg`
- `export_demo_video.py`：匯出左右對照，支援同樣的 `--stride` / `--cell-hold`

## 報告

- [8/7](PPT%20report/報告8_7.pdf)
- [7/24](PPT%20report/報告7_24.pdf)
- [7/10](PPT%20report/報告7_10.pdf)
