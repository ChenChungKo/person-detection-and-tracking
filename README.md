# person-detection-and-tracking

人物偵測、追蹤與空間定位專案（Tapo C230 / YOLO / ByteTrack / Homography）。

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

## OpenCV RTSP 即時測試

確認攝影機 RTSP 可在 VLC 播放後，於已啟動 `.venv` 的終端機執行：

```powershell
python test_rtsp.py "rtsp://帳號:密碼@攝影機IP:554/stream1"
```

- 會開啟預覽視窗（預設縮放寬度 ≤ 1280，仍讀取完整 2880×1620）
- 按 `q` 結束
- 可選參數：`--max-width 1600` 調整預覽大小
- 無視窗只測取流：`python test_rtsp.py "rtsp://..." --no-preview --frames 60`

也可先設定環境變數，避免把帳密寫進指令歷史：

```powershell
$env:RTSP_URL = "rtsp://帳號:密碼@攝影機IP:554/stream1"
python test_rtsp.py
```

## Calibration（測試中 / WIP）

> 狀態：**手動點選校正仍在測試**，流程與介面可能再調整。

用地板對應點計算 Homography（地磚參考 45 cm）。影像點需落在地板平面；原點若被遮擋可不點，改用相對座標即可。

```powershell
python calibrate_homography.py
python calibrate_homography.py "test/static_frame.jpg"
```

操作：
- 左鍵點地板點（建議 6～8 點）
- `u` 撤銷、`r` 重設、`c` 結束點選並輸入世界座標（cm）
- 預覽俯視圖後按 `s` 存成 `calibration/homography.json`，`q` 離開

測試影像：`test/static_frame.jpg`

## 文件

- [今日報告 7/10](PPT%20report/報告7_10.pdf)
