# person-detection-and-tracking

人物偵測與空間定位專案（Tapo C230 / YOLO26 / Homography）。

## 系統架構圖

![系統架構圖](picture/架構圖.png)

## 預計進度

![預計進度](picture/時程圖.png)

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

```powershell
$env:RTSP_URL = "rtsp://帳號:密碼@攝影機IP:554/stream1"
python test_rtsp.py
```

## Calibration

目前定案校正檔：`calibration/homography.json`  
- 虛擬左上角為 `(0,0)`（點不到也沒關係）  
- 有效地板區約 `X 170~530 cm`、`Y 0~540 cm`（左側桌區遮擋）  
- 地磚：左側第一格 35 cm，其餘 45 cm  

重跑／驗證：

```powershell
python calibrate_boundary.py --width 530 --height 540
python verify_homography.py
```

## 平面格子佔用

格子刻度見 `test/floor_grid_generated.jpg`（參考手繪：`test/floor_grid.png`）。

```powershell
python grid_occupancy.py
python grid_occupancy.py --x 215 --y 360
```

監視器點選地板 → 對應格子點亮。  
圖例：黃＝佔用；淺灰＝桌區／低可信（`X < 170 cm`）。

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
```

RTSP 即時已可連線並降延遲（最新幀），但**即時格子定位準度仍待修正**（影片可用）：

```powershell
python detect_grid.py --source "rtsp://帳號:密碼@攝影機IP:554/stream1" --ref auto
```

- 畫面：人框 + 腳點；超出範圍才標 `OUT`
- 格子視窗：有偵測到人時右上角顯示 `detect`／`locate` 耗時
- 按 `q` 結束，`s` 存圖  

測試影片：`test/test.mp4`

## Demo 影片（上：偵測，下：格子）

由 `test/test.mp4` 匯出的合成結果（上監視器畫面、下平面格子）：

![Demo：上偵測、下格子](test/demo_detect_grid.webp)

## 狀態備註（2026-07-15）

- 已完成：YOLO26 偵測、影片腳點／格子定位、格子 UI、RTSP 取流與延遲改善  
- 未完成／暫緩：多人 ID 追蹤、外貌 Re-ID、即時 RTSP 定位準度調校  

## 文件

- [今日報告 7/10](PPT%20report/報告7_10.pdf)
