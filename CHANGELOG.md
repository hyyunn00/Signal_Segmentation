# CHANGELOG

記錄本專案（`Signal_Segmentation`）相對於原始碼來源 `Chulab-Signal_Segmentation-main`（快照複製，無 git 歷史）的所有改動。每一項改動都依照 [改造計劃書.md](./notes/IMPROVEMENT_PLAN.md) 的稽核結果進行，格式為：**改動** / **原因** / **效果**。

基準版本：`72642d2` "Import baseline from Chulab-Signal_Segmentation-main snapshot"（未經任何修改的原始碼複製）。

---

## P0 — 量測可信化

目的：在動任何訓練/推論邏輯之前，先讓「這個改動有沒有變好」這件事可以被信任地量出來。

### 新增 `analysis3d.py`：3D 物件計數與質心距離配對

- **改動**：新增獨立腳本，把同一資料夾內的所有 z 切片 `.tif` 依檔名排序疊成單一 3D volume，用 `skimage.measure.label(volume, connectivity=3)` 做一次 26-連通標記；物件配對從 IoU 改成**質心距離配對**（容差 = 該 GT 物件的等效半徑）。像素級指標（confusion matrix、mcc、hausdorff、cldice、pr_auc）也在完整 3D volume 上算一次，不再逐片算完再平均。`--compare-2d` flag 可額外呼叫舊版 `analysis.py::evaluate_triplet()` 算出舊的逐片 2D 計數，兩者並列輸出、附上 `count_inflation_ratio`。`analysis.py` 本身的計算邏輯未被更動（只抽出 `discover_triplets()` 與泛化 `write_results()` 供兩支腳本共用）。
- **原因**：原本的 `analysis.py::evaluate_triplet()` 對每個 z 切片各自呼叫 `compute_object_metrics()`，一顆跨越多個切片的 3D 物件會被逐片各數一次，導致 `gt_count`/`pr_count` 嚴重灌水（改造計劃書.md §3.4 實測 7.1 倍）。IoU 配對對只有幾個體素寬的小物件也過於嚴格（邊界差一兩個體素就判漏檢）。
- **效果**：可以先在完全不碰訓練/分析主流程的情況下，單獨量出「舊方法 2D 計數 vs. 新方法 3D 計數」的膨脹倍率，之後再決定要不要把 pipeline 整個換掉。物件計數、detection F1、計數 MAPE 這幾個計畫書列為「主要指標」的數字，第一次有一個可信的量測方式。

### 新增 `utils/seeding.py`，接入 `train.py`：隨機性控制

- **改動**：新增 `set_global_seed(seed, deterministic=False)`，統一設定 `random`/`numpy`/`torch`（含 CUDA）與 `PYTHONHASHSEED`；`deterministic=True` 時才切到 `cudnn.deterministic=True`/`benchmark=False`，預設維持 `cudnn.benchmark=True`。`train.py` 在讀完 config、建立 dataset 之前立刻呼叫這個函式；兩個 `DataLoader` 加上 `worker_init_fn=seed_worker` 與 `generator=torch.Generator().manual_seed(seed)`。
- **原因**：原始碼完全沒有設定任何 RNG（`train.py` 無 seed 設定、DataLoader 無 `worker_init_fn`/`generator`、config 裡的 `"seed": 42` 只有傳給資料切分用）。同一份 config 跑兩次結果不同，讓 24 組（4 種初始化 × 6 個標註量）標註效率曲線實驗完全無法比較——尤其在 N=1、2 的極小標註量區間，隨機波動可能大於初始化本身的影響。
- **效果**：同一份 config 現在可重現。`deterministic` 拆成獨立旗標，一般生產訓練仍可用 `cudnn.benchmark=True` 吃滿 A6000 算力，只有在需要嚴格可重現性比較的實驗（如標註效率曲線）才犧牲速度換確定性。

### `IO/datasets.py`：train/val 改成 volume-level 切分

- **改動**：`PatchMetadata` 新增 `source_volume` 欄位；`from_folders()` 為每個 volume 抽出的所有 patch 標記其來源 volume index；`split()` 改成先對「不重複的 volume id」打亂、依 `val_ratio` 切出保留 volume 集合，同一 volume 的所有 patch 一起進 train 或一起進 val。只有 1 個可辨識 volume 時會退回舊的 patch-level 洗牌，並記警告提醒驗證數字可能偏樂觀。
- **原因**：原本 `split()` 是把所有 volume 的 patch 併成一個扁平清單後直接打亂取前 20%。標註 patch 的取樣網格彼此有重疊/相鄰的候選位置，同一 volume 裡空間相近的 patch 有機會一個進 train、一個進 val，等於用練習題的相鄰區塊當作考卷——模型只要記憶就能考高分，驗證損失/指標被系統性高估。
- **效果**：驗證集裡的每個 patch，現在保證來自一個訓練時完全沒見過任何 patch 的 volume。以「val loss 選最佳 checkpoint」這件事不會再被資料洩漏污染。

### `train.py`：metric 例外不再被靜默吞掉

- **改動**：`train_epoch()`/`valid_epoch()` 裡的 `except Exception: pass` 改成 `except Exception as e: logger.warning(...)`。
- **原因**：某個 metric 算失敗時完全沒有任何紀錄，會穩定回報 0.0 並讓學習曲線畫成一條假的連續曲線（「儀器壞了但錶還在動」）。
- **效果**：metric 計算失敗時至少會出現在 log 裡，不會被完全吃掉、變成看不見的資料損失。

---

## P1 — 遷移基礎建設

目的：讓「跨 marker 遷移學習」這個計畫書的核心目標，從完全不存在的功能變成可運作的流程。

### 新增 `utils/checkpoint.py`：checkpoint v2 格式（state_dict + 血統 + 前處理契約）

- **改動**：新增 `save_checkpoint()`（存 `model.state_dict()` 而非整個模型物件，附帶 `role`/`parent`/`shared_init_version`（血統）、`model_type`/`model_config`（可重建模型）、`preprocess`（正規化契約）、`marker`/`task_family`/`contributing_markers`/`n_annotated_crops`（內容描述）、`commit`/`env`（可追溯性）等欄位）；`load_checkpoint_for_transfer()` 用於微調載入，會（1）檢查血統——若載入的 checkpoint `role != "shared_init"` 且未設 `allow_chained_transfer`，直接報錯，強制 hub-and-spoke 而非鏈式遷移；（2）比對 checkpoint 與目前 config 的 `preprocess` 是否一致，不一致就硬擋；（3）依 `head_prefixes` 丟棄輸出頭權重、可選依 `encoder_keep_prefixes` 截斷深層 encoder；`load_checkpoint_for_inference()` 用於推論，依 checkpoint 存的 `model_type`/`model_config` 透過 `build_model_from_config()` 重建模型後 `strict=True` 載入。
- **原因**：原本 `train.py::save_checkpoint()` 是 `torch.save(model, path)`——存的是整個模型物件（pickle）。這讓「只載入 encoder、丟掉輸出頭」「凍結特定層」這些遷移學習的核心動作完全做不到；`models/` 一旦重構或改檔名，所有舊權重直接報廢；也沒有任何機制記錄一份權重是從哪個初始化微調出來的，或訓練時用的是哪一種正規化。
- **效果**：checkpoint 現在可以被部分載入、可追溯血統、可驗證前處理一致性。這是整個遷移學習流程的地基——沒有這一步，微調、分段解凍、學習率分組全部無法實作。

### `train.py`：接上微調載入路徑、encoder/head 學習率分組、三階段分段解凍

- **改動**：原本 288 行附近的空占位註解（"Note: If loading an existing model for fine-tuning, you would use load_checkpoint(path) here."）改為實際呼叫 `load_checkpoint_for_transfer()`。新增依 `config["encoder_prefixes"]`/`head_prefixes` 把參數分成 encoder / 其他兩組，分別給不同學習率（`encoder_lr_scale`，未微調時預設 1.0＝不影響現有行為，微調時預設 0.1）。新增三階段分段解凍狀態機（`train.transfer` 區塊：`freeze_epochs` 凍結 encoder 只訓練頭、`finetune_epochs` 解凍但學習率打折、其餘 epoch 全網路更低學習率），搭配 warmup + cosine 衰減。這整套機制只在 config 設了 `pretrained_weights` 時才啟動；沒設的話 `ReduceLROnPlateau` 與單一學習率群組維持原樣。
- **原因**：計畫書明確指出「transfer learning 做不好」有一部分是字面意義——這條路徑從未被實作，是工程問題不是研究問題。而直接對隨機初始化的新頭做全網路解凍訓練，是遷移學習最常見的失敗方式：隨機頭在前幾個 batch 產生的巨大梯度會把 encoder 裡有價值的特徵沖掉，症狀就是「微調後比從零訓練還差」。
- **效果**：微調現在是一個可以真的執行的流程，而不是一段死註解。`encoder_prefixes`/`head_prefixes` 刻意要求使用者用 `scripts/inspect_model_params.py` 先驗證過的字串，而不是我用未經執行驗證的猜測寫死——填錯會被記警告並自動退回單一學習率群組,不會靜默套用到錯的參數上。

### 新增 `utils/norm_utils.py`，接入 `models/factory.py`：強制 InstanceNorm

- **改動**：新增 `enforce_instance_norm(model)`，掃過模型所有子模組，把找到的 `nn.BatchNorm{1,2,3}d` 換成對應的 `nn.InstanceNorm{1,2,3}d(affine=True)`。`models/factory.py::build_model_from_config()` 建完任何模型後統一呼叫一次。
- **原因**：BatchNorm 的 running mean/var 會把來源資料集的批次統計量（也就是「這個 marker 訊號長怎樣」的資訊）直接編碼進權重，遷移到新 marker 時等於用舊 marker 的亮度分布去污染新特徵——這是跨域遷移最經典也最隱蔽的失敗方式（沒有錯誤訊息，表現就是「遷移沒效果」）。五種模型架構裡，`VNet` 在 MONAI 內部寫死用 `BatchNorm3d`，建構參數層級無法關閉。
- **效果**：不論各架構在特定 MONAI 版本下的預設是什麼，建完的模型保證不殘留 BatchNorm，主要受影響的是 `VNet`（`UNet`/`DynUNet` 預設本來就是 instance norm，`SwinUNETR` 是 transformer 不受影響）。

### `inference.py`：支援新舊兩種 checkpoint 格式

- **改動**：`load_checkpoint()` 改為偵測格式——讀到 format_version=2 的 dict 就走 `load_checkpoint_for_inference()` 重建模型；讀到舊版整物件就維持原本 `torch.load(..., weights_only=False)`，並印出 deprecation 警告。若 config 的 `inference.preprocess` 與 checkpoint 存的 `preprocess` 不一致，記警告（不像訓練端那樣硬中止——推論是生產批次工作，不希望因為契約不符就讓整批工作中斷）。
- **原因**：`inference.py` 原本唯一支援的載入方式是 `torch.load(model_path, weights_only=False)`，只能讀整物件 checkpoint；換成新格式後若不處理相容性，會讓所有既有的已訓練權重直接失效。
- **效果**：新舊 checkpoint 都能跑推論，既有的已訓練權重不需要立刻轉檔就能繼續使用；同時新格式的 checkpoint 能在推論時抓出「訓練與推論用了不同正規化」這種本來會靜默發生的錯誤。

### 新增 `scripts/inspect_model_params.py`

- **改動**：新增小工具，吃一個 config、用 `build_model_from_config()` 建出模型、印出 `model.named_parameters()` 的每一個名稱。
- **原因**：計畫書明確警告 `head_prefixes`/`encoder_prefixes` 必須先跑過這個檢查再填，因為 MONAI 的 `UNet` 是巢狀 `Sequential`（參數名長得像 `model.model.model.0...`），憑直覺猜的前綴會靜默配對不到任何參數。
- **效果**：使用者可以在寫微調 config 之前，先對自己實際用的 `model_type` 跑一次，拿到真正的參數名稱，而不是依賴我在沒有 GPU/monai 執行環境下的猜測。

### 新增 `scripts/convert_legacy_checkpoint.py`

- **改動**：新增轉檔工具，讀入舊版整物件 `.pth`，配合一份 train config（取得 `model_type`/`model_config`/`preprocess`）與 CLI 指定的 `--role`/`--marker` 等 metadata，另存成新格式 checkpoint。轉檔前會用同一份 config 建一個新模型比對 `state_dict` key 是否一致，不一致時記警告但不中止（例如 VNet 因為 BatchNorm→InstanceNorm 轉換，key 會有差異，這是預期的，但需要使用者自行確認 config 真的對應這份權重）。
- **原因**：讓既有已訓練的權重（例如 TH marker 的既有模型）能被納入新的 checkpoint 系統，作為計畫書 §5.1 候選來源 B（既有 marker 模型）的具體實作路徑。
- **效果**：既有權重不必重新訓練，就能取得血統追蹤與部分載入的能力，未來要拿來實驗「從既有模型遷移是否可行」時可以直接用。

### 新增 `configs/config_finetune_example.json`

- **改動**：新增示範用微調設定檔，展示 `pretrained_weights`、`allow_chained_transfer`、`head_prefixes`/`encoder_prefixes`、`train.transfer` 區塊怎麼填。明確標註其中的參數前綴是示意用途，需要先跑 `inspect_model_params.py` 驗證。不動任何現有生產 config（`config_cell.json`/`config_vessel.json`）。
- **原因**：新增的微調相關欄位沒有任何範例可以參考，光看 `utils/checkpoint.py`/`train.py` 的程式碼不容易知道該怎麼組出一份完整可用的 config。
- **效果**：要開始寫第一份微調 config 時有現成模板可以改，不用從零拼湊。

### `requirements.txt`：補齊缺漏套件

- **改動**：新增 `torch`、`scipy`、`tifffile`。
- **原因**：這三個套件在程式碼裡（`train.py`/`analysis.py`/`analysis3d.py`/`utils/metrics.py` 等）都是直接 import，但原本不在清單裡——`torch`/`scipy` 恰好會透過 `monai`/`scikit-image` 的相依關係間接裝到，`tifffile` 則完全沒有，會直接 `ImportError`。
- **效果**：`pip install -r requirements.txt` 之後不會缺套件。沒有虛構版本號釘死既有版本，因為沒有可靠依據知道目前實際測試過的版本組合。

---

## 資料契約與次要修正（改造計劃書.md §6 / §8 / §3.6）

目的：這些不是阻斷項，但都是計畫書明確點名、可以獨立於 P0/P1 完成的低成本修正。

### `utils/normalization.py`：新增 `percentile` 正規化模式

- **改動**：新增 `normalize_mode: "percentile"`，做「每 volume 百分位裁切（預設 0.5–99.5）後線性映射到 [0,1]」，重用既有的 histogram/CDF 機制（`FileReader` 算好的 binned histogram）反查百分位對應的強度值，不需要對整個 volume 排序。`IO/datasets.py`/`inference.py`/`preprocess.py` 裡判斷是否需要計算 histogram 的條件都從 `method == "histogram"` 改成 `method in ("histogram", "percentile")`。
- **原因**：現有的 z-score 正規化，統計量（mean/std）是對整個 volume 算的；螢光訊號前景往往只佔 1–5% 體素，代表 mean/std 幾乎完全由背景決定——等於拿背景的尺去量前景。不同 marker 的前景佔比差異極大，使得 z-score 後的數值範圍在不同 marker 間完全不可比，直接破壞「同一個 checkpoint 給不同 marker 重用」這件事。
- **效果**：多了一個不受前景佔比影響、也不需要整個 volume 排序的正規化選項，是 hub 設計（讓同一個 shared_init 能被多個 marker 重用）能夠成立的必要條件之一。目前尚未套用到任何現有生產 config，需要另外決定要不要切換。

### `IO/datasets.py`：遮罩填補改用 `constant`

- **改動**：volume-level padding（為了讓 volume 至少能塞下一個 patch）時，影像仍照 config 的 `pad_mode`（例如 `reflect`）填補，但遮罩固定用 `constant` 填 0，不再跟著 `pad_mode` 走。
- **原因**：`reflect` 填補會把邊界物件鏡像複製一份到 padding 出來的區域，等於在遮罩裡憑空製造出物件的鏡像分身，直接灌水物件計數。
- **效果**：靠近 volume 邊界的物件不會被重複計數；`analysis3d.py` 量出來的 `gt_count`/`pr_count` 會更準。

### 新增 `scripts/detect_annotation_source.py`：標註來源檢測

- **改動**：新增獨立腳本，對每個 image/mask volume 配對，比較遮罩邊界「剛好在裡面」與「剛好在外面」的體素強度分布（`scipy.ndimage` 做侵蝕/膨脹取出邊界殼層，`sklearn.metrics.roc_auc_score` 算可分性），輸出每個 volume 的 AUC 與判讀（`likely_thresholded` / `likely_manual` / `ambiguous`）。方法完全照計畫書 §6.4 給的程式碼片段實作。
- **原因**：已知現有標註「大部分全人工，但也有跑閾值再修過的」。若一份遮罩其實是從 `intensity > T` 產生的，模型學到的規則就是「亮度超過某個值」——這是所有規則裡對雷射功率、抗體批次、曝光時間變化最敏感的一條，而且這種標註一旦混進共用初始化的訓練組成，會污染整個 hub。
- **效果**：可以在不需要任何模型、不需要跑訓練的情況下，直接對現有標註跑一次檢查，找出哪些 volume 的標註可能是閾值產生、需要在建立 hub 前排除或降權。用合成資料驗證過：乾淨的閾值遮罩 AUC=1.0，形狀導向（非純亮度）的遮罩 AUC≈0.81，與計畫書給的判讀帶（≈1.00 閾值產生、0.7–0.85 人工標註）一致。

### 新增 `scripts/fingerprint_dataset.py`：資料集 fingerprinting

- **改動**：新增獨立腳本，對每個 mask volume 做 3D 連通標記，量測物件數、前景體素佔比、等效直徑分布（mean/median/p5/p95，可選轉換成 µm）、以及基於每個物件體素座標協方差矩陣特徵值算出的 elongation（細長度）指標，並依 elongation 中位數給出 `task_family_guess`（blob / tubular）。形狀/大小統計刻意不依賴 skimage 的 ND-only regionprops 欄位（例如 `equivalent_diameter`、`inertia_tensor_eigvals`），只用跨版本穩定的 `region.area`（體素數）與 `region.coords`（體素座標）自行算，避免計畫書 §3.6 提到的 skimage 版本相依風險。
- **原因**：計畫書 §6.3 明確要求「物件尺寸、前景佔比這類參數不該由人猜，應該由流程自己量測」——這些數字會決定是否截斷 encoder 深度、detection 配對容差、loss 的權重設定、以及新 marker 該歸類到哪個 task_family。
- **效果**：新增 marker 時不需要人工目測猜測形態類別和物件大小。用合成資料驗證過核心數學：球體量出 elongation≈1.0，圓柱體量出 elongation≈8.16（正確判成 tubular），等效直徑計算誤差在 0.4% 以內。腳本本身有註明這是啟發式判斷，計畫書待決事項 #7 仍要求人工確認一次。

### `utils/stitcher.py`：拼接改用距離加權

- **改動**：新增 `_make_blend_weight()`，在每個 patch 內建立一個中心權重 1.0、往邊緣線性遞減（下限 0.1，不會歸零）的權重核；`_numba_stitch_loop()` 累加時從原本的 `weight += 1`（均勻權重）改成用這個核加權累加。
- **原因**：patch 邊界的預測通常因為周圍上下文較少而品質較低，原本的均勻權重讓邊界預測跟中心預測有一樣的話語權，導致 patch 重疊處在邊界品質不佳時被拉低，在拼接後的 volume 上產生可見的接縫。
- **效果**：patch 重疊區域現在由品質較好的中心區域預測主導。用合成資料驗證過權重核本身（中心/角落權重符合設計）以及完整 `stitch_image()` 呼叫（含 numba kernel）可以正常執行、輸出有效的二值遮罩。這是純推論端的品質改善，不影響訓練。

### 新增 `utils/transforms.py`：augmentation 改為 config 驅動

- **改動**：把原本寫死在 `train.py` 檔案頂部的 transform pipeline，搬進 `utils/transforms.py` 的 `build_train_transform(aug_config)`/`build_val_transform()`，由 `train.config["augmentation"]` 驅動（`DEFAULT_AUGMENTATION_CONFIG` 提供預設值）。同時：三軸翻轉取代原本只翻一個軸；新增連續角度旋轉（`RandAffined`）；取消原本被註解掉的 `RandGaussianNoised`；新增可選的（預設關閉）各向異性模糊，模擬較粗的 z 解析度。`train.py` 改成在 `main()` 內呼叫這兩個 builder。
- **原因**：整個專案是 config 驅動的（正規化、模型、loss、指標都是），唯獨 transform 是寫死常數，導致增強策略無法掃參、無法記錄進 artifacts、無法針對不同 marker 用不同配方。既有的增強策略本身也有三個缺口：雜訊增強被註解掉（模型沒見過訊噪比變化）、只翻一個軸（浪費掉兩個免費的不變性）、完全沒有模糊增強。各向異性模糊同時是計畫書 §9.5「階梯一」PSF 域隨機化的零成本版本。
- **效果**：增強策略現在跟其他所有超參數一樣進 config、進 artifacts 紀錄。預設值讓既有生產 config 的行為大致不變（新增的雜訊/多軸翻轉/旋轉是附加的），各向異性模糊預設關閉、需要主動開啟。⚠️ 加強增強後 in-domain 驗證損失會變差是預期行為，要看的是 `analysis3d.py` 的保留集 detection 指標，不是驗證損失。

---

## 尚未處理（需要人為決定，非純程式碼問題）

- **物理解析度對齊（spacing，§6.1）**：`IO/reader.py` 完全沒有讀取任何物理解析度 metadata。要從檔案本身讀（不確定現有 TIFF/zarr 有沒有存）或每個 volume 手動填 config，兩條路實作方式差很多，需要先決定資料來源。
- **PR-AUC 修復（§3.6）**：`utils/stitcher.py` 目前寫檔前就二值化，`compute_pr_auc` 因此恆為 0。要修就要新增一個「輸出原始機率」的模式，會讓輸出檔案從二值 uint8/16 變成儲存空間更大的浮點格式，需要先確認儲存成本可接受。
- **P2 之後**（spacing 對齊以外的資料契約項目、P3 hub v0、P4 標註效率曲線實驗、P5/P6）：需要真實標註資料或 GPU 上實際訓練，不是能在目前這個純程式碼環境解決的。
