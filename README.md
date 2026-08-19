# person-detection-and-tracking

監視器畫面做人體偵測與地板格子定位（Tapo C230 / YOLO26 / Homography）。

<p align="center">
  <img src="picture/架構圖.png" alt="系統架構圖" width="560" />
</p>

## 現在怎麼跑

請用專案虛擬環境 `C:\5Gjump\.venv`（VS Code：`Python: Select Interpreter` → `.venv`）。

```powershell
cd C:\5Gjump
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

首次偵測會下載 `yolo26s.pt`。建議指令（本機 `test4`）：

```powershell
python detect_grid.py --source test/test4.mp4 --ref pose --cell-hold 2 --quiet --reid-model osnet_ain --error-comp calibration/homography_error_report.json
```

RTSP 只要把 `--source` 換成 `rtsp://帳號:密碼@IP:554/stream1`（自動 TCP、只讀最新幀）。單人影片可改 `test/test.mp4` 並用 `--ref auto`。

| 加這個 | 作用 |
|--------|------|
| `--no-pose-skeleton` | 不畫骨架 |
| `--review-dump` | 把裁圖寫到 `test/reid_review/`（不參與即時比對） |
| `--save-video 路徑.mp4 --no-show --no-realtime` | 錄左右對照影片 |
| `--no-track` | 只要人框、不要 ID |
| `--no-floor-grid` | 關掉地上 A/B/C/O 對照點 |

按 `q` 結束，`s` 存圖。預覽寬度上限 1280。

<p align="center">
  <img src="test/demo_stable_id_osnet_ain.webp" width="100%" alt="Demo：左偵測右格子"/>
</p>

完整 mp4：`test/demo_stable_id_osnet_ain.mp4`。

## 管線

`YOLO26.track` + BoT-SORT（短期軌、yaml 不開短 ReID）→ Stable-ID + OSNet-AIN（長期 ID）→ 腳點投到地板格子。畫面與格子用同一套 ID 顏色。不假設場上只有一人。

`--ref pose` 時偵測仍用 `yolo26s.pt`，骨架另跑 `yolo26s-pose.pt`。腳點：

| 情況 | 用哪裡 |
|------|--------|
| 坐（連續坐姿證據） | 髖 X + 框底 |
| 站、腳踝可見 | 腳踝 |
| 站、下半身被擋 | 同一 ID 須先有完整站姿歷史才補腳；否則框底 |

圖例：綠＝座位、青＝腳踝、橘＝歷史補腳、紅＝框底、紫＝推估。骨架只是顯示（`--kpt-draw-conf` 預設 0.25）。

| 旗標 | 預設／建議 | 一句話 |
|------|------------|--------|
| `--ref pose` | 建議 | 見上表 |
| `--error-comp` | 建議開 | 四點卷尺 affine，修遠端 Homography |
| `--reid-model osnet_ain` | 建議 | 長期 ID |
| `--stride` | 5 | 每 5 幀跑一次 YOLO；本機影片固定取這些幀，結果可重現 |
| `--cell-hold` | 2 | 格子連續兩次偵測一致才亮／滅 |
| `--min-hits` | 16 | 新 ID 約 4 秒才發號（已登錄的人再出現不必再等） |
| `--conf` | 0.45 | 人框門檻 |

即時圖庫 `test/reid_gallery/` 每次重跑清空。貼畫面邊緣的框立刻清除；室內漏檢會短沿用。

地上對照點 A/B/C/O（cm）在 `calibration/floor_marks.json`：A(170,450) 近左、B(170,180) 遠左、C(440,450) 近右、O(260,315) 走道中心。改點：`python pick_floor_marks.py --source test/static_frame.jpg`。

## 校正（現況）

日常定位讀 `calibration/homography.json`（**v2 棋盤、原始畫面、不去畸變**），再加上 `--error-comp calibration/homography_error_report.json`。

Homography 在棋盤附近準，離板子遠（尤其近端右側）會偏。四點實測後做世界座標 affine，那 4 點上平均約 **32 cm → 6 cm**（最大約 **90 → 10 cm**）。格子約 45 cm，走道中間常常還是同一格；補償修的是 Homography 系統偏差，**不能**修桌後遮擋造成的腳點亂跳。

鏡頭內參 `calibration/camera_intrinsics.json` **有檔，但 `detect_grid.py` 不讀**。試過 undistort 再重估 H：棋盤附近略好，廣角邊緣過強，所以沒接進即時流程。重跑 `calibrate_chessboard_floor.py calibrate` 時若偵測到內參會自動去畸變——目前不要這樣做。

舊版手動 Homography：`--calib calibration/homography_v1_manual.json`。舊桌區灰格：`--valid-xmin 170`。

### 重標地板（需要時）

```powershell
# 量測誤差／更新四點報告
python verify_homography.py --measure-error --image calibration/chessboard_floor/capture.jpg

# v2 棋盤（先 capture 再 calibrate；不要讓它自動套內參）
python calibrate_chessboard_floor.py capture --source "rtsp://帳號:密碼@攝影機IP:554/stream1"
python calibrate_chessboard_floor.py calibrate --image calibration/chessboard_floor/capture.jpg --origin-x 190 --origin-y 400 --out calibration/homography_v2_chessboard.json

# v1 手動點選
python calibrate_boundary.py --width 530 --height 540
```

座標系：虛擬左上 `(0,0)`；地板約 X 0–530 cm、Y 0–540 cm；左側第一格 35 cm，其餘 45 cm。

## 已知限制

- 多人遮擋／漏檢仍可能閃號；近鏡頭大框易吃到另一人
- `test3.mp4` 交叉頻繁，`--min-hits 16` 偏保守
- 體型／衣服都很像的換人，OSNet 仍可能配回舊號
- 桌後遮擋時腳點仍可能跳格
- 開頭幾秒沒 ID 是在等 `--min-hits`，不是漏檢

## 其它腳本

- `python test_rtsp.py "rtsp://..."`：只測串流（TCP、預覽 ≤1280）
- `python detect_person.py --source ... --no-map`：只要 YOLO 人框
- `python grid_occupancy.py`：點監視器地板看格子
- `python export_demo_video.py`：匯出對照影片（同樣支援 `--stride` / `--cell-hold`）

BoT-SORT 設定在 `trackers/botsort.yaml`（靜態相機 `gmc_method: none`，`with_reid: False`）。不要把 OSNet 權重寫進 yaml。

## 報告

- [8/7](PPT%20report/報告8_7.pdf)
- [7/24](PPT%20report/報告7_24.pdf)
- [7/10](PPT%20report/報告7_10.pdf)
