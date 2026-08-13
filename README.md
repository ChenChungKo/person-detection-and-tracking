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
測試影片：`test/test.mp4`。按 `q` 結束，`s` 存圖。

### 人物 ID（Stable-ID）

管線：`YOLO26` → **ByteTrack**（短追蹤）→ **Stable-ID**（gallery 外貌 + 地板距離接回；同幀不共用 ID）。

| 項目 | 說明 |
|------|------|
| Re-ID | `--reid-model osnet_ain`（建議）／`osnet`／`osnet_ibn`／YOLO26-ReID ONNX |
| Gallery | `test/reid_gallery/ID00x/`（`first`／`latest`；雙人近距離時可能暫不寫圖） |
| 發新號 | 須連續失敗 gallery（`--min-hits` 預設 8）才 mint，減少亂發 ID3／ID4 |
| 顯示 | 畫面只標 `ID1`…；人框／ID 標籤與格子佔用**同一套 ID 顏色**；`--id-coast` 預設短，避免人走後框殘留 |

```powershell
# 建議（單人／test.mp4 較穩）
python detect_grid.py --source test/test.mp4 --ref auto --cell-hold 2 --quiet --reid-model osnet_ain --stride 2 --log-id

# RTSP
$env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"
python detect_grid.py --source "rtsp://帳號:密碼@IP:554/stream1" --ref auto --cell-hold 2 --quiet --reid-model osnet_ain --stride 2
```

BoT-SORT（`--tracker trackers/botsort_reid.yaml`）短追蹤只用 YOLO-ReID ONNX；Stable-ID 仍可用 `osnet_ain`。不要把 OSNet `.pth` 寫進 BoT-SORT yaml。

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

## 目前建議設定（2026-08-06）

| 項目 | 設定 |
|------|------|
| Homography | **v2**（`calibration/homography.json`） |
| 偵測 | `yolo26s.pt`、`--ref auto`、`--conf 0.50`、`--cell-hold 2` |
| ID | ByteTrack + Stable-ID；`--reid-model osnet_ain`；`--min-hits 8`；`--appear-thresh 0.34` |
| 效能 | `--stride 2`（OSNet 時建議）；跳幀框外推；本機影片即時跟播 |

```powershell
python detect_grid.py --source test/test.mp4 --ref auto --cell-hold 2 --quiet --reid-model osnet_ain --stride 2 --log-id
```

舊校正：`--calib calibration/homography_v1_manual.json`；舊桌區灰格：`--valid-xmin 170`。

## Demo 影片（左：偵測，右：格子）

**目前主 demo（Stable-ID + OSNet-AIN）**，由 `test/test.mp4` 匯出：

```powershell
python export_demo_video.py --source test/test.mp4 --ref auto --cell-hold 2 --reid-model osnet_ain --stride 2 --out test/demo_stable_id_osnet_ain.mp4
```

| 版本 | 影片 | WebP |
|------|------|------|
| **Stable-ID + OSNet-AIN**（目前） | `test/demo_stable_id_osnet_ain.mp4` | `test/demo_stable_id_osnet_ain.webp` |
| Homography **v2**（舊，僅定位） | `test/demo_v2_chessboard.mp4` | `test/demo_v2_chessboard.webp` |
| Homography **v1** | `test/demo_v1_manual.mp4` | `test/demo_v1_manual.webp` |

<p align="center">
  <img src="test/demo_stable_id_osnet_ain.webp" width="100%" alt="Demo：Stable-ID + OSNet-AIN，左偵測右格子"/>
</p>

## 狀態（2026-08-06）

**已完成**
- YOLO26 偵測 + Homography v1／v2 腳點格子定位、RTSP 降延遲、跳幀、格子防抖  
- Stable-ID：gallery 外貌接回、多外貌 prototype、OSNet／OSNet-AIN／OSNet-IBN  
- 圖庫防混人、較短 box coast（人離開後框不長留）  
- 不同 ID 在即時畫面人框／標籤與格子佔用使用同一套顏色  
- 單人／`test.mp4`：ID 大致穩定可用  

**尚未解決**
- 雙人／多人近距離：仍可能漏检造成 `ID1,ID2 ↔ ID2` 閃爍  
- 近鏡頭大框易吃到另一人 → 圖庫可能拒寫或外貌混淆  
- 換裝、跨鏡頭、畸變校正接入管線  

## 文件

- [8/7](PPT%20report/報告8_7.pdf)
- [7/24](PPT%20report/報告7_24.pdf)
- [7/10](PPT%20report/報告7_10.pdf)
