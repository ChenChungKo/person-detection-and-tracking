# person-detection-and-tracking

Tapo C230 監視器：偵測人、給穩定 ID、把腳點投到預先規劃的地板格子。原理與限制請參考 [說明書.md](說明書.md)。

![系統架構圖](picture/架構圖.png)

日常預設：棋盤 Homography v2、四點補償、YOLO26s + pose、OSNet-AIN。  
即時管線不要去畸變。ChArUco 與手動點磚都不要直接覆蓋 `calibration/homography.json`。不要把 OSNet 寫進 `trackers/botsort.yaml`。

## demo影片

示範是左監視器、右格子。主片：[test4_review.mp4](test/review_videos/test4_review.mp4)

![Demo：左監視器、右格子](test/review_videos/test4_review.webp)

舊版對照：[棋盤 v2](test/demo_v2_chessboard.mp4)、[手動點磚 v1](test/demo_v1_manual.mp4)。

---

## 1. 安裝環境

用專案虛擬環境。VS Code 選 Interpreter → `.venv`。

```powershell
cd C:\5Gjump
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

之後可以跑下面的腳本。第一次偵測會下載 `yolo26s.pt` 與 `yolo26s-pose.pt`。

---

## 2. 規劃地板格子

格子是**先在世界座標量好、切好**，不是從監視器畫面裁出來的。換廠、換機也要先量這間房的地板。虛擬左上角 `(0, 0)`（被擋住點不到也沒關係）；Y 朝相機增大。

開 GUI，填總長、總寬、切幾格。可勾「頭尾較小，中間等分」。本場預設：X 530 cm、12 格、頭格 35 cm；Y 540 cm、12 格全等分。按預覽看圖，按儲存。

```powershell
python make_floor_grid.py
```

![規劃地板格子 GUI](test/demo_floor_grid_gui.jpg)

得到格子圖 `test/floor_grid_generated.jpg`，切法寫入 `calibration/floor_grid.json`（之後偵測與格子視窗用這份）。

---

## 3. 舖板、校正 Homography（三種）

板子平放在地上、完整入鏡。原點是編號內角 **#0**（本場約 190, 400）；紙張近左外角約 170, 420。

先寫到新檔，確認後再複製成即時用的 `calibration/homography.json`。三種互不覆蓋。**日常用棋盤。** 不要套鏡頭去畸變。

### 3.1 棋盤（日常）

自動抓角點。偵測 GUI 的校正選「棋盤」。影像預設就是剛拍的 `calibration/chessboard_floor/capture.jpg`。

拍一張：

```powershell
python calibrate_chessboard_floor.py capture --source "rtsp://帳號:密碼@IP:554/stream1"
```

用原點 (190, 400) 計算，先存新檔：

```powershell
python calibrate_chessboard_floor.py calibrate --origin-x 190 --origin-y 400 --out calibration/homography_v2_chessboard.json
```

角點圖上的原點 #0 位置正確後，再覆蓋偵測實際讀的 `homography.json`：

```powershell
Copy-Item calibration\homography_v2_chessboard.json calibration\homography.json
```

![棋盤角點與原點 #0](calibration/chessboard_floor/origin_marked.jpg)

### 3.2 ChArUco（試驗）

交點帶編號。只寫 `homography_charuco.json`，不取代棋盤。偵測 GUI 的校正選「ChArUco」才會用這份。補償用 `calibration/charuco_floor/error_report.json`，不要用棋盤那份（第 5 步）。

```powershell
python calibrate_charuco_floor.py --capture
```

得到 `calibration/homography_charuco.json`。

![ChArUco 角點與原點 #0](calibration/charuco_floor/detected_corners.jpg)

### 3.3 手動點磁磚角（舊法）

在畫面上點看得見的地磚角，輸入該點的世界座標。誤差較大，不要當日常預設。

```powershell
python calibrate_boundary.py --width 530 --height 540 --out calibration/homography_v1_manual.json
```

得到 `homography_v1_manual.json`。

![手動點選磁磚角](calibration/v1_tile_clicks.jpg)

---

## 4. 地上對照點

Homography 做好之後，才在畫面上點看得見的地面，寫入 `calibration/floor_marks.json`。偵測畫面與格子會標這四點。


| 點   | 世界座標 (cm)  | 位置   |
| --- | ---------- | ---- |
| A   | (170, 450) | 近左   |
| B   | (170, 180) | 遠左   |
| C   | (440, 450) | 近右   |
| O   | (260, 315) | 走道中心 |


依序點 A → B → C → O，按 `s` 存檔：

```powershell
python pick_floor_marks.py --source test/static_frame.jpg
```

![地上對照點 A B C O](test/floor_overlay_cam.jpg)

在監視器地板上點一下，看右邊格子亮哪一格：

```powershell
python grid_occupancy.py
```

不點畫面，直接用世界座標 (215, 360) 公分看亮哪一格：

```powershell
python grid_occupancy.py --x 215 --y 360
```

---

## 5. 驗證定位、四點補償

點地板看預測的世界座標（公分）：

```powershell
python verify_homography.py
python verify_homography.py --error-comp calibration/homography_error_report.json
```

棋盤附近準、離板子遠（尤其近端右側）會偏。在地上量 4 個已知點，存成補償檔；GUI「四點補償」與 `--error-comp` 用這份。不要把 v2 的補償套到 ChArUco。

```powershell
python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg
```

得到 `calibration/homography_error_report.json`。

![四點實測補償](test/error_comp_four_points.jpg)

---

## 6. 開偵測

```powershell
python launch_detect_grid.py
```

按 Run 後左格子、右監視器。預設是本機影片、校正棋盤、腳點 pose、YOLO `yolo26s`、Re-ID `osnet_ain`，並勾選骨架、追蹤／ID、四點補償、A/B/C/O。

可調的項目：

- 來源：本機影片或 RTSP
- 校正：棋盤、ChArUco
- 腳點：pose、auto、foot、head_drop
- 勾選：骨架、追蹤／ID、四點補償、A/B/C/O、少印 log、審查裁圖
- YOLO：yolo26s、yolo26m、yolo26l
- Re-ID：osnet_ain、osnet_ibn、osnet、none
- 拉桿：conf、stride、cell-hold、min-hits、out-margin

![detect_grid 執行中：五人都在格子與監視器](test/demo_detect_five.jpg)

本機多人片 `test/test4.mp4`（或指令如下）。按 `q` 結束、`s` 存圖。示範：`test/review_videos/test4_review.mp4`。

```powershell
python detect_grid.py --source test/test4.mp4 --ref pose --cell-hold 2 --quiet --reid-model osnet_ain --error-comp calibration/homography_error_report.json
```

即時：GUI 選 RTSP，網址 `rtsp://帳號:密碼@IP:554/stream1`（本場 `rtsp://oriongo:123456789@192.168.0.200:554/stream1`）。自動 TCP、只處理最新幀。只測串流：

```powershell
python test_rtsp.py "rtsp://帳號:密碼@攝影機IP:554/stream1"
```

單人片可用 `test/test.mp4`，腳點改 `auto`。

---

## 說明與報告

- 完整說明：[說明書.md](說明書.md)
- [9/18](PPT%20report/報告9_18.pdf)（[PPT](PPT%20report/報告9_18.pptx)）
- [8/21](PPT%20report/報告8_21.pdf)（[PPT](PPT%20report/報告8_21.pptx)）
- [8/7](PPT%20report/報告8_7.pdf)（[PPT](PPT%20report/報告8_7.pptx)）
- [7/24](PPT%20report/報告7_24.pdf)（[PPT](PPT%20report/報告7_24.pptx)）
- [7/10](PPT%20report/報告7_10.pdf)（[PPT](PPT%20report/報告7_10.pptx)）

