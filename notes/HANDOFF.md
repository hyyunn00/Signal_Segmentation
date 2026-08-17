# 交接筆記（給接手這個專案的 Claude session）

寫給另一台電腦上接手的 Claude Code session。這份文件的目的是讓你不用重新讀完整個對話紀錄，就能知道現在在哪、下一步要做什麼、哪些決定已經定案不要重新問。

## 這是什麼

`Signal_Segmentation` 是根據 [`notes/IMPROVEMENT_PLAN.md`](./IMPROVEMENT_PLAN.md)（原檔名「改造計劃書.md」）在做的跨 marker 遷移學習改造。原始程式碼快照複製自上一層資料夾的 `Chulab-Signal_Segmentation-main`（無 git 歷史，是稽核時的原樣快照——不要跟另一個有後續 commit 的 `Chulab-Signal_Segmentation` 搞混）。所有改動都在 `Signal_Segmentation` 這個 repo 裡進行，原始碼快照不動。

GitHub remote：`https://github.com/hyyunn00/Signal_Segmentation.git`（`main` 分支，已推上去，開始接手前先 `git pull` 確認同步）。

**執行環境**：實際訓練/推論會在 Linux server（A6000 GPU + 512GB RAM）上跑。目前為止的開發都在沒有 `torch`/`monai`/GPU 的本機做的，所有新增的數值/checkpoint 相關程式碼都只做過 `py_compile`/`pyflakes` 靜態檢查，加上少數用臨時裝的 `numpy`/`scipy`/`scikit-learn`/`scikit-image`/`numba`（不含 `torch`/`monai`）做的合成資料 smoke test——**還沒有在真正的訓練環境跑過一次端到端流程**。這是接手後第一批要做的事之一。

## 現在的狀態（已完成並 push 到 `origin/main`）

依 commit 順序：

1. `72642d2` baseline import（原始碼快照，未修改）
2. `1b39bc0` **P0 量測可信化**：`analysis3d.py`（3D 連通標記 + 質心距離配對，取代逐 2D 切片計數；`--compare-2d` 可量出新舊計數膨脹倍率）、`utils/seeding.py`（全域 RNG 控制，`deterministic` 旗標拆開速度/可重現性）、`IO/datasets.py` volume-level train/val 切分（取代 patch-level、避免資料洩漏）、`train.py` metric 例外從靜默吞掉改成有 log。
3. `bb54228` **P1 遷移基礎建設**：`utils/checkpoint.py`（checkpoint v2：`state_dict` + 血統欄位 `role`/`parent`/`shared_init_version` + 前處理契約，取代整物件 `torch.save`）、`train.py` 接上微調載入路徑 + encoder/head 學習率分組 + 三階段分段解凍（只在設了 `pretrained_weights` 時啟動，不影響現有從零訓練）、`utils/norm_utils.py` 強制 InstanceNorm、`inference.py` 支援新舊 checkpoint 格式、`scripts/inspect_model_params.py`（印參數名，驗證 `encoder_prefixes`/`head_prefixes` 用）、`scripts/convert_legacy_checkpoint.py`（舊權重轉新格式）、`configs/config_finetune_example.json` 示範設定。
4. `3655375` **§6/§8/§3.6 低成本項目**：`utils/normalization.py` 新增 `percentile` 正規化模式、`IO/datasets.py` 遮罩填補改用 `constant`（不再跟著影像用 `reflect`）、`scripts/detect_annotation_source.py`（標註來源檢測，閾值產生 vs 人工標註）、`scripts/fingerprint_dataset.py`（物件尺寸/前景佔比/形狀量測）、`utils/stitcher.py` 拼接改距離加權（取代均勻權重）、`utils/transforms.py` 增強改 config 驅動（三軸翻轉、連續角度旋轉、恢復雜訊增強、可選各向異性模糊）。
5. `d020c6c` `CHANGELOG.md`——**每一項改動的「改動/原因/效果」完整記錄，這是你應該先讀的檔案**，比這份交接筆記細很多。
6. `b6fba01` `改造計劃書.md` 搬到 `notes/IMPROVEMENT_PLAN.md`；新增 `notes/LITERATURE_REFERENCES.md`（文獻整理筆記）。

## 明確定案的決定（不要重新問、不要重新提起）

**arXiv:2605.26026 不整合進計畫書。** 這篇論文（一個 3D light-sheet 螢光顯微 foundation model，架構含 SwinUNETR、前處理用百分位正規化，跟本專案有些技術上的相似）曾被提議列為 `shared_init v0` 的候選，**使用者已明確兩次拒絕**（一次直接拒絕、一次在看過更完整的文獻整理後仍維持拒絕）。`notes/LITERATURE_REFERENCES.md` 裡雖然整理了這篇論文的細節，但那份文件**只當背景參考，不代表要採納**。如果又看到有人提議把這篇論文排進 `shared_init` 候選來源，先跟使用者確認過，不要自己動手改 `IMPROVEMENT_PLAN.md` 的 §5.1/§5.2/§10。

**`IMPROVEMENT_PLAN.md` §12 待決事項，已回答的部分：**

| # | 問題 | 答案 | 意涵 |
|---|---|---|---|
| 3 | 各 marker 取像條件是否相同？ | 不同——**要做 spacing 對齊** | §6.1 的物理解析度對齊要動手做了，不是純理論討論 |
| 4 | 磁碟餘裕？ | **512 GB** | §6.1 建議的方案①（上採樣至較細者）需要這個數字夠不夠撐；接手後第一件事是估算現有資料集總大小，確認 512GB 夠不夠用方案①，不夠的話退回方案③ |
| 5 | 成像模態？ | **光片（light-sheet），但使用者用「光片吧」回答，不是 100% 確定** | §9.2 PSF 路線的適用性前提基本成立，但不要當作絕對確定——§9 本身優先序仍然低，不用因為這個答案就急著動 §9 |
| 6 | 標註是 instance 還是單一二值遮罩？ | **單一二值遮罩**，不是逐物件 instance | §2.4 blob head 的「質心熱圖」不能直接從標註讀出來，要先對二值遮罩做連通元件標記才能得到每個物件的質心——之後做 blob head 時要記得這一步 |

**另外兩項先前討論但不在正式待決事項表格裡的決定：**

| 項目 | 答案 | 意涵 |
|---|---|---|
| spacing 的來源要從檔案 metadata 讀，還是手動填 config？ | **手動填 config** | 不用去研究 TIFF/zarr 檔案裡有沒有存 spacing metadata（`IO/reader.py` 目前完全沒有這個概念）。實作方向：在 config 加一個 `target_spacing_um`／每個 volume 的來源 spacing 由使用者手動填，不要做自動偵測 |
| PR-AUC 修復要不要接受推論輸出檔案變大（存原始機率而非只存二值遮罩）？ | **好，接受** | `utils/stitcher.py` 目前寫檔前就二值化，導致 `compute_pr_auc` 恆為 0（見 `IMPROVEMENT_PLAN.md` §3.6）。可以動手加一個「輸出原始機率」的模式了 |

## 建議下一步（照這個順序做，理由見下方）

### 第一批：純程式碼工作，不需要真實資料就能開始

1. **Spacing 對齊**（§6.1，已確認要做，來源手動填 config）：新增一個 resample 工具（放 `utils/` 底下，例如 `utils/spacing.py`），讀 config 裡人工填的 `target_spacing_um` 與每個 volume 的原始 spacing（也是人工填的，因為讀不到 metadata），做重採樣。**影像用線性插值（`order=1`），遮罩必須用最近鄰（`order=0`）**——這是計畫書自己強調的重點，線性插值會讓遮罩出現非二元值、侵蝕小物件。接上 `IO/datasets.py`/`preprocess.py` 的前處理流程。checkpoint 的 `preprocess` 契約（`utils/checkpoint.py`）也要把 `target_spacing_um` 納入比對。
2. **PR-AUC 修復**（§3.6，已確認接受檔案變大）：`utils/stitcher.py::stitch_image()` 目前在 `_numba_finalize_reconstruction()` 裡直接二值化再回傳。改成可以選擇輸出原始機率（浮點或高位元 uint16 尺度化），`inference.py` 的輸出設定加一個開關（例如 `output.save_probabilities: true`），預設關閉、不影響現有行為。

這兩項不需要真實資料，可以直接動手；跟 P0/P1 一樣走「不影響既有行為的預設值 + 明確 opt-in」的原則，不要動到現有生產 config（`configs/config_cell.json`/`config_vessel.json`）。

### 第二批：需要真正的訓練環境（Linux server，torch/monai/GPU）才能跑

3. `scripts/inspect_model_params.py --config <實際會用的 config>`——對每個 `model_type` 跑一次，拿到真正的 `encoder_prefixes`/`head_prefixes`。**這是任何微調嘗試的前提**，P1 寫的分段解凍/學習率分組在沒有這步之前都是空的。
4. `analysis3d.py --compare-2d`——拿現有已訓練模型的推論輸出跑，不需要重新訓練。這是 `IMPROVEMENT_PLAN.md` §3.4 自己指定的「建議第一個動作」。
5. `scripts/fingerprint_dataset.py`——對真實 mask volume 跑，直接把 §12 #7（各 marker 典型物件直徑）從「需人工確認」變成有數字可看，順便看每個 marker 該歸 blob 還是 tubular。
6. `scripts/detect_annotation_source.py`——對真實 image/mask 配對跑，檢查有沒有標註其實是閾值產生的。

7. **上面兩批都做完、且 §12 剩下的 #1/#2（哪個 marker 標註最多、每個 marker 幾隻動物）也問到答案後**，才進入 `IMPROVEMENT_PLAN.md` 路線圖的 P3（hub v0，用 Cellpose 或自監督預訓練，**不是** arXiv:2605.26026——見上面「明確定案的決定」）。

## 關鍵檔案地圖

| 要找什麼 | 去哪 |
|---|---|
| 每項改動的完整說明（改動/原因/效果） | `CHANGELOG.md` |
| 原始技術稽核與計畫全文 | `notes/IMPROVEMENT_PLAN.md` |
| 文獻整理（含 arXiv:2605.26026 細節，僅供參考） | `notes/LITERATURE_REFERENCES.md` |
| checkpoint 存讀、血統檢查、前處理契約 | `utils/checkpoint.py` |
| 微調載入、學習率分組、分段解凍 | `train.py`（搜尋 `pretrained_path`） |
| 3D 評估、2D vs 3D 診斷 | `analysis3d.py` |
| 隨機性控制 | `utils/seeding.py` |
| 拼接（含新的距離加權） | `utils/stitcher.py` |
| 正規化（含新的 percentile 模式） | `utils/normalization.py` |
| 增強策略（改 config 驅動後） | `utils/transforms.py` |
| 微調 config 範例 | `configs/config_finetune_example.json` |

## 協作慣例（使用者對之前工作方式的回饋，照著做）

- **程式碼註解一律用英文**，即使溝通/文件用中文。
- **不要猜測參數名稱去寫死**（例如 encoder/head 的 state_dict key 前綴）——寫一支驗證工具讓使用者自己跑出來確認（`scripts/inspect_model_params.py` 就是這樣來的），不要憑印象猜。MONAI 的 `UNet` 是巢狀 `Sequential`，猜錯會靜默失敗。
- **不動現有生產 config**（`config_cell.json`/`config_vessel.json`）——新功能一律用「預設值 = 現有行為不變，opt-in 才啟用新行為」的方式加，不要求使用者被迫遷移。
- **不做無關的機械翻譯或大規模重構**——例如不會把既有中文註解整段翻英文，只有新寫/修改的部分才用英文。
- 大改動前用 plan mode 先跟使用者對過再動手；exploratory 問題（"接下來能做什麼"這類）先給簡短建議，不要直接開始實作。
- Commit 訊息用英文，說明改動與原因；只有使用者明確要求才 push 到 remote。
- 本機沒有 GPU/torch/monai 時，誠實講清楚「只做過語法檢查/合成資料 smoke test，沒有端到端驗證過」，不要宣稱測試過沒測過的東西。
