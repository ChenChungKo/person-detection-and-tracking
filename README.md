# person-detection-and-tracking

人物偵測與空間定位專案（Tapo C230 / YOLO26 / Homography）。

## 系統架構圖

<p align="center">
  <img src="picture/架構圖.png" alt="系統架構圖" width="560" />
</p>

## 預計進度

<p align="center">
  <img src="picture/時程圖.png" alt="預計進度" width="560" />
</p>

## 環境建置

```powershell
cd C:\5Gjump
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

在 VS Code 選擇解譯器：`Python: Select Interpreter` → `.venv`

首次執行偵測時會自動下載 `yolo26s.pt`（也可改用 `yolo26n.pt` / `yolo26m.pt`）。

## OpenCV RTSP 即時測試

```powershell
python test_rtsp.py "rtsp://帳號:密碼@攝影機IP:554/stream1"
```

- 預覽預設寬度 ≤ 1280（仍讀取完整 2880×1620）
- 按 `q` 結束
- 無視窗：`python test_rtsp.py "rtsp://..." --no-preview --frames 60`
- 腳本會將 RTSP 設為 **TCP** 傳輸（較穩）

```powershell
$env:RTSP_URL = "rtsp://帳號:密碼@攝影機IP:554/stream1"
python test_rtsp.py
```

完整偵測流程的降延遲說明見下方「RTSP 降延遲」。
## Calibration

地板 Homography 有兩個版本（都保留，互不覆蓋）：

| 版本 | 檔案 | 作法 | 備註 |
|------|------|------|------|
| **v1** | `calibration/homography_v1_manual.json` | 手動點選磁磚角（`calibrate_boundary.py`） | 點擊量測誤差約 **8.9 cm** |
| **v2** | `calibration/homography_v2_chessboard.json` | 地板大棋盤格自動角點（`calibrate_chessboard_floor.py`） | 目前預設；點擊量測誤差約 **3.8 cm** |

預設腳本讀 `calibration/homography.json`（目前＝**v2**）。要比對 v1 時加上 `--calib`：

```powershell
python detect_grid.py --source test/test.mp4 --calib calibration/homography_v1_manual.json
python detect_grid.py --source test/test.mp4 --calib calibration/homography_v2_chessboard.json
```

座標系：
- 虛擬左上角為 `(0,0)`（點不到也沒關係）  
- 地板格範圍約 `X 0~530 cm`、`Y 0~540 cm`  
- 地磚：左側第一格 35 cm，其餘 45 cm  
- 預設**不再**把左側畫成淺灰桌區（`--valid-xmin 0`）；若要恢復舊遮罩：`--valid-xmin 170` 

重跑／驗證：

```powershell
# v1 手動點選
python calibrate_boundary.py --width 530 --height 540
python verify_homography.py

# v2 棋盤格（先 capture 再 calibrate）
python calibrate_chessboard_floor.py capture --source "rtsp://帳號:密碼@攝影機IP:554/stream1"
python calibrate_chessboard_floor.py calibrate --image calibration/chessboard_floor/capture.jpg --origin-x 190 --origin-y 400 --out calibration/homography_v2_chessboard.json
```

真實定位誤差（點擊已知地板點 + 輸入卷尺座標，不是棋盤擬合殘差）：

```powershell
python verify_homography.py --measure-error
python verify_homography.py --measure-error --calib calibration/homography_v1_manual.json
python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg
```

結束後會印平均／最大誤差，並可存 `calibration/homography_error_report.json`。

## 平面格子佔用

格子刻度見 `test/floor_grid_generated.jpg`（參考手繪：`test/floor_grid.png`）。

```powershell
python grid_occupancy.py
python grid_occupancy.py --x 215 --y 360
```

監視器點選地板 → 對應格子點亮。  
圖例：黃＝佔用；若加 `--valid-xmin 170`，淺灰＝桌區／低可信（`X < 170 cm`）。

## YOLO 人框測試

預設模型：`yolo26s.pt`。

```powershell
python detect_person.py --source test/test.mp4 --no-map
python detect_person.py --source "rtsp://帳號:密碼@攝影機IP:554/stream1" --no-map
```

## 偵測 + 定位（腳點 → 格子）

以 bbox 底邊中點為腳點；桌旁被擋時可用 `--ref auto` / `--ref head_drop`。

**目前建議先用本機影片驗證定位**（較穩）：

```powershell
python detect_grid.py --source test/test.mp4 --ref auto
python detect_grid.py --source test/test.mp4 --ref foot
python detect_grid.py --source test/test.mp4 --no-track --stride 3
```

RTSP 即時範例：

```powershell
# 偵測 + 定位 + ID（預設）
python detect_grid.py --source "rtsp://帳號:密碼@攝影機IP:554/stream1" --ref auto --cell-hold 2

# 只要定位、可跳幀加速
python detect_grid.py --source "rtsp://帳號:密碼@攝影機IP:554/stream1" --ref auto --no-track --stride 3 --cell-hold 2
```

- 畫面：人框 + 腳點 + **僅顯示 `ID`**（預設開追蹤；`--no-track` 關閉）；超出範圍才標 `ID OUT`
- `OUT`：腳點世界座標遠超格子才標；預設允許 **45 cm** 邊界容差（`--out-margin`，約一格），避免 Homography／腳點誤差在邊緣誤判  
- 格子視窗：右上角固定顯示上次 `detect`／`locate` 耗時（沒偵測到人也會保留上一次數值，不會消失／閃爍）
- 按 `q` 結束，`s` 存圖  
- **即時格子定位準度仍待修正**（本機影片較穩）

測試影片：`test/test.mp4`

### 人物 ID 追蹤（預設開啟）

管線：`YOLO26 detect` → **ByteTrack**（短時關聯）→ **穩定 ID 層**（ByteTrack 換號後，用「可走到的地板距離 + 外貌相似」決定是否接回舊 ID）。

- **不預設假設畫面只有同一人**；`--single-person` 僅供 demo，預設關閉  
- 畫面標籤**只顯示 `ID1` / `ID2`…**（格子座標改寫在 console）  
- **誤檢抑制**：人框幾何過濾（拒螢幕／過短框）、`conf` 預設 0.50、新目標需連續 `--min-hits 3` 次才發 ID／顯示（避免一閃誤檢佔號）  
- **ID 不回收**：號碼只往上加；離開的人靠地板距離＋外貌接回原 ID（同一程式執行期間記住）  
- **外貌**：預設 **YOLO26 Re-ID**（`yolo26n-reid.onnx`，首次自動下載）；`--no-reid` 才退回 HSV 衣服顏色  
- Tracker 設定：`trackers/bytetrack_stable.yaml`（可改 `--tracker trackers/botsort_reid.yaml` 讓短時追蹤也用 Re-ID）  
- 本機 `.mp4` 預設**即時跟播**（推論慢會丟幀）；要逐幀慢播加 `--no-realtime`  
- 關閉追蹤：`--no-track`

```powershell
python detect_grid.py --source test/test.mp4 --ref auto --cell-hold 2 --quiet
# RTSP
python detect_grid.py --source "rtsp://帳號:密碼@IP:554/stream1" --ref auto --cell-hold 2 --quiet
# 較強 Re-ID / 短時追蹤也開 Re-ID
python detect_grid.py --source test/test.mp4 --reid-model yolo26s-reid.onnx --tracker trackers/botsort_reid.yaml --quiet
# 退回舊的 HSV 顏色
python detect_grid.py --source test/test.mp4 --no-reid --quiet
```

常用參數：`--reid-model`、`--appear-thresh`、`--conf`、`--min-hits`、`--out-margin`。

換裝／極端光照仍可能失敗；更重的跨鏡頭 Re-ID 訓練可視需求再加。

### RTSP 降延遲（自動啟用）

來源為 `rtsp://` 時，`detect_grid.py` / `detect_person.py` 會自動做兩件事（實作見 `latest_frame.py`）：

1. **只處理最新幀（`LatestFrameCapture`）**  
   YOLO 推論較慢時，OpenCV/FFmpeg 會把攝影機新幀堆在緩衝區；若依序 `cap.read()`，畫面會落後數秒。背景執行緒持續讀流並**只保留最新一幀**（舊幀直接覆蓋丟棄），主執行緒每次推論都拿「當下最新畫面」，優先保證即時性（中間幀會被捨棄）。

2. **RTSP 走 TCP**  
   開串流前設定 `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`，比 UDP 穩、較少因封包遺失造成卡頓或重連。

這與下方 `--stride` 不同：

| 機制 | 目的 | 作用 |
|------|------|------|
| `LatestFrameCapture` | 降低「畫面落後感」 | 丟緩衝區舊幀，永遠處理最新畫面 |
| `--stride` | 降低運算量 | 不必每幀都跑 YOLO |

本機 `.mp4` 不會啟用最新幀讀取（逐幀播放比較合理）。

### 跳幀（`--stride`）

RTSP／影片不必每幀都跑 YOLO，可跳幀降低運算量，中間幀沿用上次偵測結果：

```powershell
python detect_grid.py --source test/test.mp4 --stride 3
```

- `--stride N`：每 N 幀才跑一次 YOLO（**預設 2**；設 1＝每幀都跑）
- 格子視窗會標示 `cached`，代表這幀是沿用結果、不是新偵測
- 本機影片另有 `--realtime`（預設開）：依影片時間軸丟幀，避免播放變慢動作

### 格子防抖（`--cell-hold`）

站著不動時，因 bbox 微小晃動（例如身體扭動）經 Homography 放大，偶爾會讓判定的格子跳到隔壁格造成閃爍。加上防抖：

```powershell
python detect_grid.py --source test/test.mp4 --cell-hold 2
```

- 格子需連續 N 次偵測結果一致才會點亮／熄滅（預設 2；設 1 等於關閉防抖）
- 這裡的「N 次」以**偵測次數**計算（跟 `--stride` 搭配時，只算真正跑 YOLO 的那幀，不受跳幀影響其穩定邏輯）
- `export_demo_video.py` 也支援同名的 `--stride` / `--cell-hold`

## 目前最佳設定（2026-07-31）

經 RTSP／影片實測後，**目前建議預設組合**：

| 項目 | 設定 |
|------|------|
| Homography | **v2** 棋盤格（`calibration/homography.json`＝v2） |
| 格子範圍 | **全格**（`--valid-xmin 0`，不畫左側桌區灰格） |
| 偵測 | `yolo26s.pt`、`--ref auto`、`--conf 0.50` |
| 追蹤 | **ByteTrack** + 地板距離／**YOLO26 Re-ID** 接回 ID；`--min-hits 3` |
| 即時／效能 | 預設 `--stride 2` + 本機影片即時丟幀；關追蹤可 `--stride 3` |

建議指令：

```powershell
# 偵測 + 定位 + ID（較順：跳幀＋即時跟播）
python detect_grid.py --source test/test.mp4 --ref auto --cell-hold 2 --quiet

# 只要定位、不要 ID
python detect_grid.py --source test/test.mp4 --ref auto --no-track --stride 3 --cell-hold 2
python detect_grid.py --source "rtsp://帳號:密碼@攝影機IP:554/stream1" --ref auto --no-track --stride 3 --cell-hold 2
```

舊版手動校正仍保留：`--calib calibration/homography_v1_manual.json`；舊桌區灰格：`--valid-xmin 170`。

## Demo 影片（左：偵測，右：格子）

皆由同一支 `test/test.mp4` 匯出（`--stride 3 --cell-hold 2 --valid-xmin 0`），僅 Homography 版本不同：

| 版本 | 影片 | WebP |
|------|------|------|
| **v1** 手動點選 | `test/demo_v1_manual.mp4` | `test/demo_v1_manual.webp` |
| **v2** 棋盤格（目前預設／最佳） | `test/demo_v2_chessboard.mp4` | `test/demo_v2_chessboard.webp` |

README 主圖使用 **v2**：

<p align="center">
  <img src="test/demo_v2_chessboard.webp" width="100%" alt="Demo v2：左偵測、右格子"/>
</p>

v1 對照：

<p align="center">
  <img src="test/demo_v1_manual.webp" width="100%" alt="Demo v1：左偵測、右格子"/>
</p>

（舊檔 `demo_detect_grid.webp` 仍保留，內容等同當時的預設 demo。）

## 狀態備註（2026-07-31）

- 已完成：YOLO26 偵測、**ByteTrack + 地板距離／YOLO26 Re-ID 接回 ID**（預設不假設單人、ID 不回收、畫面只顯示 ID）、誤檢過濾、影片／RTSP 腳點格子定位、Homography **v1（手動）／v2（棋盤格）**、全格顯示、RTSP 降延遲、跳幀、本機即時跟播、格子防抖、OUT 容差、點擊量測真實誤差  
- 目前最佳：見上方「目前最佳設定」  
- 未完成／暫緩：換裝／跨鏡頭專用 Re-ID 微調、鏡頭畸變校正接入管線  

## 文件

- [7/24](PPT%20report/報告7_24.pdf)
- [7/10](PPT%20report/報告7_10.pdf)
