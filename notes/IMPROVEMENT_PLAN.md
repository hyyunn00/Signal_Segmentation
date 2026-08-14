# 跨 Marker 遷移學習　技術稽核與改造計劃書

**標的**：`Chulab-AdvancedMicroscopyLab/Chulab-Signal_Segmentation`
**目標**：建立一組可重複使用的共用初始化權重與微調流程，使新增一種螢光 marker 時只需少量標註即可訓練出可用模型，不必從零開始。

---

## 1. 目標定義

### 1.1 要做的

建立單一的共用初始化 `shared_init.pth`。新增 marker 時從它出發微調，得到該 marker 專屬的權重。

```
                    ┌──────────────────┐
                    │  shared_init.pth │   ← 唯一的起點，版本化
                    └────────┬─────────┘
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   cFos_v3.pth          GCaMP_v1.pth         Lectin_v2.pth
```

每個 marker 的權重獨立演化，互不干擾。

### 1.2 明確不做

- 單一模型在推論時同時處理多個 marker
- 推論時的 marker 自動辨識
- 跨 marker 的持續學習與遺忘控制

因為每個 marker 各有一組權重，**災難性遺忘不是本計劃需要處理的問題**。這讓設計比一般的持續學習系統簡單很多——不需要 adapter 模組、不需要經驗回放、不需要遺忘控制機制。

### 1.3 成功的定義

不是「Dice 提高幾分」，而是：

> **達到可接受品質所需的標註量減少多少？**

若從零訓練需要 40 個標註 crop 才能達到 detection F1 0.85，而從共用初始化出發只需要 8 個，那就是 5 倍的標註效率提升。

**這是一筆重複發生的成本，不是一次性的。** 現有 marker 累積了多少標註，與本計劃的價值無關——因為**每一個新進的 marker 都是從零開始**。實驗室只要還會繼續加入新的訊號類型，這條流程的效益就會隨每次新增而重複兌現。

除了標註量，遷移還帶來三項不隨標註量消失的效益：**收斂更快**（省 GPU 時數）、**低學習率下更穩定**、**最終分數通常小幅較好**。

### 1.4 一個重要區分：共同訓練的兩種用途

| 用途 | 是否採用 |
|---|---|
| **生產共用初始化**——把多個 marker 的標註混合訓練出一個通用起點 | ✅ 採用 |
| **部署單一模型**——一個模型在推論時同時處理所有 marker | ❌ 不採用 |

這兩者不衝突。實驗室累積的多種 marker 標註（c-Fos、TH、Lectin、CD31⋯）正是製造好初始化所需要的異質語料，目前完全沒被用在這件事上。

---

## 2. 架構決策

### 2.1 Hub-and-spoke，不做鏈式遷移

所有 marker 權重都必須是 `shared_init` 的**直接子代**，血統深度永遠為 1。禁止 A→B→C 的鏈式遷移。

四個理由：

**不累積偏誤。** 鏈式遷移下，C 繼承了 B 的所有怪癖，而 B 又繼承了 A 的。三代之後沒有人知道模型為什麼對某種形狀特別敏感。

**可追溯。** Hub 之下，任何權重的來歷都是 `shared_init vX → marker`，只有一個東西需要版控。Pairwise 之下血統是一張有向圖，半年後「你當初從哪個模型開始的」會變成沒人答得出來的問題。

**實驗可歸因。** 標註效率曲線需要固定的起點。若每個 marker 從不同祖先出發，曲線之間的差異無法解讀。

**起點本身更好。** 用多個 marker 訓練出來的 hub，比任何單一 marker 模型都更通用。單一 marker 模型會把該 marker 的歸納偏置烘進去——在管狀結構上訓練的 encoder 會對細長物體過度反應，拿去做團塊任務時容易把血管、組織裂縫、自體螢光纖維誤判為前景。

**實作上的紀律**：checkpoint 記錄 `role` 欄位；若 `pretrained_weights` 指向的是 `marker_specific` 而非 `shared_init`，訓練腳本發出警告並要求明確確認。

### 2.2 Encoder 共用，head 分家

**不要為不同的形態類別（task family）各做一個 backbone。**

```
shared_init.pth
├── encoder          ← 一份，所有 family 共用
└── heads/
    ├── blob.pth      （前景 + 質心熱圖）
    └── tubular.pth   （前景 + 骨架 / 距離場）
```

**理由一：資料飢餓程度不同。** Encoder 是最吃資料量、最需要多樣性的部分。切成兩半等於各自減半，會從「有點通用」變成「兩個各自狹窄」。而 encoder 學的低中階特徵——邊緣、斑點、雜訊統計、局部對比——本來就跟形態類別無關，遷移價值主要就在這裡。

Head 相反：它是 family 特異性所在，但重訓成本很低。該分的是這裡。

**理由二：形態是連續的，不是二元的。** TH 在同一張影像裡就同時有胞體（團塊）與纖維（管狀）。神經突起、星狀膠質細胞突起、樹突都落在中間。分成兩個 backbone 之後，這類 marker 沒有位置放。

**理由三：維護成本。** 兩個產物、兩條重訓流程、兩份版本史，交接時要交兩個東西而不是一個。

### 2.3 Encoder 用完整深度，由 decoder 決定用幾層

這裡有一個必須處理的技術衝突：**小團塊物件適合淺網路，管狀結構需要深網路。**

物件在網路中逐層縮小。若某 marker 的物件只有約 5 個體素寬：

```
stride 2  → 2.5
stride 4  → 1.25
stride 8  → 0.6    ← 已小於一個體素
stride 16 → 0.3
```

深層對這種物件不再有定位資訊。但管狀結構相反——需要大感受野才能學到「這一段和那一段是同一條血管」，砍深度會讓血管斷成一節一節。

若深度不同，encoder 就不是同一個形狀，權重無法共用。

**解法：encoder 統一用完整深度。**

深層特徵對小團塊沒有幫助，但只要 skip connection 保留高解析度路徑，它也**不會有害**——代價是計算量與參數量，不是準確度。為了維持單一產物付這個代價是划算的。

而且因為存的是 `state_dict`，微調時可以**只載入前面幾個 block、把深層丟掉**，等於免費得到一個淺版本。這正是整物件序列化做不到的事——hub 設計與 state_dict 是互相成全的。

是否需要截斷，由標註效率曲線決定：同一個 hub，一組載入全部、一組截斷到 stride 8，比較兩條曲線。

### 2.4 形態類別的定義

| | **團塊類（blob）** | **管狀類（tubular）** |
|---|---|---|
| 例 | c-Fos、GCaMP 胞體、細胞核 | Lectin、CD31、神經纖維 |
| 科學終點 | 數量、逐細胞強度 | 密度、連通性、管徑、長度 |
| 輸出頭 | 前景 + 質心熱圖 | 前景 + 骨架 / 距離場 |
| 損失 | Dice + Focal | Tversky + clDice（拓樸感知） |
| 主要指標 | detection F1、計數 MAPE | clDice、連通元件數、骨架長度誤差 |
| patch 形狀 | 近等向 | 可非等向 |

現有 config 已支援 per-task 的 loss 與 metrics（`config_cell` 用 `dice+focal` 與 `iou`，`config_vessel` 用 `tversky+focal` 與 `bce`，patch 為 `[32,64,64]`）。要新增的是 `task_family` 這一層抽象，讓**輸出頭、指標、配對規則**一起跟著切換，而不是只有 loss 各自設定。

---

## 3. 阻斷項稽核

以下項目不修正，遷移學習無法進行或無法驗證。所列程式碼行為皆已逐檔確認。

### 3.1　checkpoint 序列化整個模型物件　🔴

```python
# train.py
torch.save(model, path)

# inference.py
return torch.load(model_path, weights_only=False)
```

這等於把整台機器連同說明書用膠水封起來。想只取出引擎裝到別的車上——做不到。

具體後果：

1. **無法部分載入權重。** 跨 marker 遷移的核心動作就是「載入 encoder、丟棄輸出頭」，這在整物件序列化下不可能。2.3 節的「只載入前幾個 block」也做不到。
2. **無法凍結特定層。** 分段解凍策略無法實施。
3. **`models/` 一旦重構或改檔名，所有舊權重報廢**（pickle 記錄的是類別的完整匯入路徑）。
4. 跨 PyTorch 版本容易損壞；`weights_only=False` 有安全性顧慮。

**這是整個計劃中效益最高的單一改動，也是所有後續工作的前提。**

### 3.2　沒有微調路徑　🔴

`train.py` 第 270 行附近：

```python
# Note: If loading an existing model for fine-tuning, you would use load_checkpoint(path) here.
```

註解下方是空的。每次執行都是 `build_model_from_config()` 隨機初始化開始。

**「transfer learning 做不好」有一部分是字面意義上的——這條路徑從未被實作。** 這是工程問題，不是研究問題。

### 3.3　訓練／驗證切分以 patch 為單位　🔴

`IO/datasets.py` 的 `from_folders()` 把所有 volume 的 patch 以 `torch.cat` 併成單一張量，`split()` 對這個扁平清單隨機排列後取前 20%。

這等於把習題冊撕成碎片、隨機抓兩成當考卷。考卷上的題目與練習題來自同一塊組織，模型只要記憶就能考高分。

**對本計劃特別致命的原因**：標註效率曲線的價值在於比較不同初始化在**小標註量**下的差異，而小標註量正是資料洩漏影響最大的區間。1 個 crop 訓練、驗證卻來自同一個 crop 的相鄰區塊，各條曲線會全部黏在一起，實驗做了也讀不出結論。

**修正**：以動物個體（或至少 volume）為單位分組切分。

### 3.4　評估是逐 2D 切片進行的　🔴

`analysis.py` 的 `evaluate_triplet()` 對資料夾中每個 `.tif` 檔各自 `imread` 後呼叫 `compute_object_metrics()`。輸出為 Scroll-Tiff（每個 z 平面一個檔案），因此實際上是在每一張二維切片上做連通元件標記。

一顆橘子橫切五片，逐片去數，就數成五顆橘子。

報表中的 `gt_count`、`pr_count`、`obj_f1` 全都是二維截面層級的統計。合成資料實測：54 顆三維物件，逐片統計得到 384 個「物件」，誇大 7.1 倍（實際倍率取決於物件的 z 軸厚度）。

**一個必要的區分**：這與 3.3 是兩個不同的缺陷。訓練時的驗證曲線被資料洩漏污染；而 `analysis.py` 若跑在真正沒進訓練的 volume 上，那是合法的保留集評估，只是計數方式錯誤。若對外報告的數字來自 `analysis.py`，被高估的主要是計數倍率而非洩漏。兩者要分開檢視。

**修正**：`analysis3d.py` 已提供——堆疊成 3D volume 後做 26-連通標記，並以質心距離配對取代 IoU 配對。

⚠️ **建議這是第一個動作**：先只放這支腳本、不改任何訓練程式碼，用現有權重重跑一次評估。這樣可在動 pipeline 之前，單獨量到「二維逐片 vs 三維真實」的差距。這個差距與模型好壞無關，純粹是量測方式問題。

### 3.5　隨機性未受控　🔴

- `train.py` 沒有任何 seed 設定
- DataLoader 無 `worker_init_fn` 與 `generator`
- `utils/cropper.py` 的 `filter_indices_by_mask()` 使用兩次全域 `np.random.shuffle()`
- config 的 `"seed": 42` 只傳給 `split()`

同樣設定跑兩次結果不同。標註效率曲線要比較 4 種初始化 × 6 個標註量 = 24 組訓練，若基準線本身在浮動，所有差異都無法歸因。

在 N = 1 或 2 的極小標註量區間，隨機波動的影響甚至可能大於初始化的影響。**這一項必須在實驗開始前修好，且每個設定要跑 3 個 seed 取中位數。**

### 3.6　次級但會干擾判讀的項目

| 項目 | 位置 | 問題 |
|---|---|---|
| 指標例外被靜默吞掉 | `train.py` ×2、`utils/metrics.py` | 儀器壞了但錶還在動。失效的指標穩定回報 0.0，並因曲線延續邏輯而畫出假的連續學習曲線 |
| 以驗證損失選模型 | `train.py` | `dice + 0.2×focal` 與「數對幾顆物件」不同步 |
| 物件配對用 IoU ≥ 0.5 | `utils/metrics.py` | 對小物件過嚴，邊界差一兩個體素就判定漏檢 |
| PR-AUC 恆為 0 | `utils/stitcher.py` | 輸出在寫檔前已二值化，`compute_pr_auc` 直接回傳 0；連帶無法事後調閾值、無法估不確定度 |
| 遮罩用 reflect 填補 | `IO/datasets.py` | 邊界物件被鏡像複製，計數灌水 |
| 正樣本判定只需 1 個體素 | `utils/cropper.py` | 一個雜訊體素就讓整個 64³ 區塊進正樣本 |
| 拼接使用均勻權重 | `utils/stitcher.py` | patch 邊界的低品質預測與中心等權重，產生可見接縫 |
| 相依套件未鎖版 | `requirements.txt` | MONAI 的 `SwinUNETR` 建構參數跨版本有變；`scipy` 未列入 |

---

## 4. 遷移機制設計

### 4.1　哪些層可以遷移

| 層次 | 學到什麼 | 跨 marker 可遷移性 |
|---|---|---|
| 淺層 encoder | 邊緣、斑點、雜訊統計、局部對比 | ✅ 幾乎完全可用——與 marker 無關 |
| 深層 encoder | 形狀先驗、上下文、「什麼構成一個物件」 | 🟡 部分可用——取決於形態是否相近 |
| decoder | 如何從特徵還原邊界 | 🟡 部分可用 |
| 輸出頭 | 這個 marker 具體長什麼樣 | ❌ 必須重學 |

實務配方：**載入 encoder（必要時連 decoder），重新初始化輸出頭，分段解凍。**

### 4.2　checkpoint 規格

```python
torch.save({
    "format_version": 2,
    "state_dict": model.state_dict(),

    # ---- 血統紀律（強制 hub-and-spoke）----
    "role": "shared_init",              # shared_init | marker_specific
    "parent": None,                     # marker_specific 必須指向某個 shared_init
    "shared_init_version": "v1",

    # ---- 重建模型所需 ----
    "model_type": model_type,
    "model_config": model_config,
    "in_channels": 1,
    "encoder_depth": 4,                 # 供 2.3 的截斷載入使用

    # ---- 內容描述 ----
    "marker": None,                     # shared_init 為 None；marker 權重才填
    "task_family": None,                # shared_init 為 None；head 才有
    "contributing_markers": ["c-Fos", "Lectin", "CD31"],   # hub 用了哪些資料
    "n_annotated_crops": 137,

    # ---- 前處理契約（見 6）----
    "preprocess": {
        "normalize_mode": "percentile",
        "percentiles": [0.5, 99.5],
        "target_spacing_um": [1.8, 1.8, 2.0],
    },

    # ---- 可追溯性 ----
    "seed": 42,
    "commit": git_hash,
    "env": {"torch": ..., "monai": ...},
}, path)
```

三段設計說明：

**`role` / `parent` / `shared_init_version`** 是 2.1 節血統紀律的實作。載入時檢查：若 `pretrained_weights` 的 `role` 是 `marker_specific`，發出警告並要求 config 明確設定 `allow_chained_transfer: true` 才繼續。這樣不會有人不小心開始鏈式遷移。

**`preprocess`** 特別重要。遷移學習最常見的隱性失敗，就是訓練時用一組正規化、推論時用另一組。把它綁進 checkpoint，載入時比對不符即中止，可完全消除這類錯誤。

**`contributing_markers`** 記錄 hub 是用哪些資料造出來的。當某個 marker 的遷移效果特別差時，這是第一個要查的東西——很可能該 marker 的形態在 hub 的訓練組成中完全沒有代表。

### 4.3　載入與分段解凍

```python
pretrained = config.get("pretrained_weights")
if pretrained:
    ck = torch.load(pretrained, map_location="cpu")

    # 血統檢查
    if ck.get("role") != "shared_init" and not config.get("allow_chained_transfer", False):
        raise ValueError(
            f"pretrained_weights 的 role='{ck.get('role')}'（marker={ck.get('marker')}）。"
            f" 本專案採 hub-and-spoke，應從 shared_init 出發。"
            f" 若確定要鏈式遷移，設定 allow_chained_transfer=true。")

    # 前處理契約檢查
    if ck["preprocess"] != config["preprocess"]:
        raise ValueError(f"preprocess mismatch:\n ckpt={ck['preprocess']}\n cfg={config['preprocess']}")

    sd = ck["state_dict"]

    # 丟棄輸出頭，讓它重新初始化
    if config.get("reset_head", True):
        sd = {k: v for k, v in sd.items()
              if not any(k.startswith(p) for p in config["head_prefixes"])}

    # 可選：截斷深層 encoder（2.3 節）
    if config.get("truncate_encoder_to"):
        sd = _truncate(sd, config["truncate_encoder_to"])

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[transfer] from={pretrained} init_version={ck.get('shared_init_version')} "
                f"loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
```

⚠️ `head_prefixes` 必須依實際模型填寫。**先跑一次 `print([n for n, _ in model.named_parameters()])` 確認。** MONAI 的 `UNet` 是巢狀 `Sequential`（`model.model.model.0...`），無法用直覺的前綴選取；`SwinUNETR` 的 encoder 在 `model.model.swinViT.*`。

**三階段排程**：

| 階段 | epochs | encoder | decoder / head | 學習率 | 目的 |
|---|---|---|---|---|---|
| 1 | ~20 | 凍結 | 訓練 | 1e-3 | 讓新的頭快速對齊既有特徵 |
| 2 | ~60 | 解凍，LR × 0.1 | 訓練 | 1e-4 | 特徵微調 |
| 3（選用） | ~30 | 全網路 | 全網路 | 1e-5 | 收斂精修 |

**第一階段不可省略。** 直接全網路解凍時，隨機初始化的輸出頭會產生巨大梯度，把 encoder 裡有價值的特徵沖掉。這是遷移學習最常見的失敗方式，症狀是「微調後比從零訓練還差」——很多人遇到這個就結論「遷移學習沒用」。

### 4.4　學習率分組

```python
enc, dec = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (enc if _is_encoder(n) else dec).append(p)

optimizer = optim.AdamW([
    {"params": enc, "lr": lr * config.get("encoder_lr_scale", 0.1)},
    {"params": dec, "lr": lr},
], weight_decay=wd)
```

搭配線性 warmup（約 5 epoch）加 cosine 衰減。小標註量下 warmup 特別重要。

### 4.5　正規化層的選擇會決定遷移成敗

**必須使用 InstanceNorm，不要用 BatchNorm。**

BatchNorm 會把來源資料的批次統計量（running mean / var）硬編碼進權重。載入到新 marker 時，這些統計量描述的是**另一種訊號的亮度分布**，會直接污染特徵。這是跨域與跨任務遷移最經典的失敗原因，而且很隱蔽——表現出來就是「遷移沒效果」，不會有錯誤訊息。

InstanceNorm 逐樣本正規化，不攜帶資料集層級的統計量，天然適合遷移。目前 `models/` 使用 MONAI 預設，需明確指定：

```python
norm_name=("INSTANCE", {"affine": True})
```

**額外好處**：若日後想做極輕量的適應，可以只微調 InstanceNorm 的仿射參數（僅數百至數千個參數，幾分鐘收斂，因自由度極低而不可能過擬合）。

---

## 5. shared_init 的來源與版本策略

### 5.1　候選來源

| 來源 | 成本 | 預期效果 | 備註 |
|---|---|---|---|
| **A. 隨機初始化** | 0 | 基準線 | 現況 |
| **B. 既有 marker 模型** | 0（已存在） | 中 | 最直接，但會帶入該 marker 的歸納偏置。**在 hub 設計下只適合當 v0 的暫代品，不是長期方案** |
| **C. 細胞基礎模型** | 低 | 可能最高 | Cellpose、StarDist、micro-sam 已在大量異質細胞影像上預訓練。對團塊類任務很可能優於實驗室自有的幾個 volume 訓練結果。版本更新快，建議查證最新狀態 |
| **D. 自監督預訓練** | 中（需算力，不需標註） | 高 | 用實驗室**現有大量無標註影像**做 MAE / SimMIM。這批資料目前完全閒置 |
| **E. 多 marker 共同訓練** | 中 | 高 | 把所有既有標註混合訓練。這是製造 hub 最直接的方式 |

⚠️ **不要預設 B 一定可行。** 「從既有模型遷移」這個假設本身就該被測試。C 與 D 很可能更好，成本未必更高。

### 5.2　版本策略：先跑流程，再最佳化內容

**`shared_init v0` 不需要解決 family 問題。** 直接用 Cellpose 權重，或用無標註資料的自監督預訓練結果即可——這兩個都不需要先想清楚形態類別怎麼切。

Family-specific head 是 v1 或 v2 的事。**先把 hub 這個流程跑起來，比先把 hub 的內容做到最好重要**——前者是基礎建設，後者可以持續改進。

| 版本 | 內容 | 前置條件 |
|---|---|---|
| **v0** | Cellpose 或自監督預訓練的 encoder，單一通用 head | 第 3 節阻斷項修畢 |
| **v1** | 多 marker 共同訓練的 encoder + blob / tubular 兩個 head | 有 ≥ 3 個 marker 的標註可用 |
| **v2** | 加入新累積的標註重訓，涵蓋更多形態 | 標註庫成長後定期執行 |

每次發布新版本，用同一套標註效率曲線驗證新版確實優於舊版，再切換預設。

---

## 6. 資料契約

要讓一組權重能被任意 marker 重複使用，輸入必須先被規範到同一個「語言」。

| 項目 | 規格 | 理由 |
|---|---|---|
| 通道 | 單通道 | marker 不進通道維度；一組權重對應一種 marker |
| 強度 | **每 volume 的百分位裁切**（0.5–99.5）後映射至 [0,1] | 見 6.2 |
| 物理尺度 | 統一 `target_spacing_um` | 見 6.1 |
| 極性 | 一律亮訊號在暗背景 | 螢光影像天然符合 |

### 6.1　物理解析度對齊

**現況**：`preprocess.py`、`IO/reader.py`、`converter.py` 全部只處理格式與強度，**沒有任何 spacing 概念**。config 的 `training_resize_factor` 固定為 `[1,1,1]` 且是單純比例縮放。patch size 以體素定義。

**為什麼影響遷移**：一顆直徑 10 µm 的物件，在 z = 2 µm 的資料中跨越 5 個切片、呈現為近似球體；在 z = 4 µm 中只跨越 2.5 個切片、呈現為扁盤。**同一個物件對模型而言是兩種不同形狀。**

一組權重要能被不同來源的資料重複使用，前提是所有資料都被規範到同一個物理尺度。**沒有這一步，checkpoint 的可重用性無從談起。**

| 方案 | 目標 z | 優點 | 代價 |
|---|---|---|---|
| ① 上採樣至較細者 | 較小值 | 不丟棄真實量測；近等向使物件呈球狀 | 資料量增加 |
| ② 下採樣至較粗者 | 較大值 | 儲存最省 | 永久丟棄解析度；物件過薄時 z 方向難以分離相鄰個體 |
| ③ 折衷 | 中間值 | 儲存衝擊小 | 兩邊都有插值 |

建議 ①，需先確認磁碟餘裕。實作要點：影像用線性插值（`order=1`），**遮罩必須用最近鄰（`order=0`）**——線性插值會產生非二元值並侵蝕小物件。

### 6.2　正規化

**現況**：預設 `z-score`，統計量由整個 volume 計算（**每個 volume 各自計算，並非跨資料集共用**）。

**問題**：螢光訊號稀疏，前景可能只佔 1–5% 體素，因此平均值與標準差**幾乎完全由背景決定**——等於拿背景的尺去量前景。

不同 marker 的前景佔比差異極大（細胞核稀疏、血管網絡密集），這使得 z-score 後的數值範圍在不同 marker 之間完全不可比，**直接破壞遷移**。

百分位裁切不受前景佔比影響，是 hub 設計的必要條件。

⚠️ **注意**：`low_cut` / `high_cut` 是**絕對強度值**的裁切，不是百分位。沒有現成的百分位機制可用，需新寫。

### 6.3　dataset fingerprinting

既然這套要服務任意 marker，**物件尺寸、前景佔比這類參數不該由人猜，應該由流程自己量測**。

```python
def fingerprint(mask_volumes, spacing_um):
    """訓練前自動執行，結果寫進 artifacts 與 checkpoint。

    回傳：
      - 物件體積分布、等效直徑（µm 與 voxel）
      - 前景體素佔比
      - 每 volume 物件數
      - 形狀指標（球度、細長度）
    """
```

自動導出的設定：

| 量測值 | 決定 |
|---|---|
| 等效直徑（voxel） | 是否截斷 encoder 深度（2.3 節） |
| 等效直徑 | detection 配對容差（取半徑） |
| 前景佔比 | Tversky 的 α/β、`neg_keep_ratio`、focal 權重 |
| 形狀指標 | `task_family`，進而決定輸出頭與指標 |
| spacing | 重採樣目標 |

這樣新增 marker 時不需人工調參，設定來源可追溯，且**新 marker 的 family 歸屬由量測決定而非人為判斷**。

### 6.4　標註來源檢測

已知標註「大部分全人工，但也有跑閾值再修過的」。這一項值得單獨查，可能是重大發現。

**為什麼**：若遮罩是從 `intensity > T` 產生的，模型學到的就是「亮度超過某個值」這條規則。這是所有規則裡**對條件變化最敏感的一條**——換雷射功率、換抗體批次、換曝光時間，立刻失效。

**閾值衍生的標註，本身就是在教模型一件無法遷移的事。** 這些標註若進入 hub 的訓練組成，會污染整個共用初始化。

**檢測方法**（一個下午可完成）：比較遮罩邊界「剛好在裡面」與「剛好在外面」的體素強度分布。

```python
from scipy import ndimage
from sklearn.metrics import roc_auc_score

def threshold_signature(image, mask, shell=1):
    m = mask > 0.5
    inner = m & ~ndimage.binary_erosion(m, iterations=shell)
    outer = ndimage.binary_dilation(m, iterations=shell) & ~m
    vals = np.concatenate([image[inner], image[outer]])
    lab  = np.concatenate([np.ones(inner.sum()), np.zeros(outer.sum())])
    return roc_auc_score(lab, vals)
    # ≈1.00    → 邊界完全由單一強度決定 = 閾值產生
    # 0.7–0.85 → 人工標註（人會依形狀與脈絡判斷，不只看亮度）
```

**後續動作**：
1. 把 `annotation_method` 記入 metadata
2. 比較模型在「純人工標註」與「閾值標註」volume 上的表現落差
3. 決定是否從 hub 的訓練組成中排除或降權

⚠️ 若日後要做標註者一致性測試當作效能天花板，必須**分開統計兩類標註**——混在一起量到的是「人與閾值法的一致性」，不是「人與人」。

### 6.5　遮罩填補

`IO/datasets.py` 目前對影像與遮罩使用相同的 `pad_mode`（config 為 `reflect`）。影像鏡像填補合理，但**遮罩鏡像會在體積邊界外憑空製造物件的鏡像分身**，直接灌水計數。遮罩應改為 `constant` 填 0。

---

## 7. 核心實驗：標註效率曲線

本計劃最重要的交付物。

### 設計

- **固定測試集**：一組完全保留、以動物為單位隔離的目標 marker volume
- **變數 1**：訓練標註量 N ∈ {1, 2, 5, 10, 20, 50} 個 crop
- **變數 2**：初始化來源 ∈ {隨機, 既有 marker 模型, 基礎模型, 自監督 / 共同訓練 hub}
- **重複**：每個組合 3 個 seed，取中位數並標出全距
- **輸出**：x 軸標註量（對數刻度），y 軸 detection F1，各初始化一條曲線

### 回答三個問題

1. **遷移省下多少標註**——達到相同 F1 所需標註量的比值
2. **哪個 hub 來源最好**——決定 5.1 節要投資哪一條
3. **飽和點在哪**——超過某個標註量後曲線收斂，代表遷移不再值得投資

### 曲線形狀與取樣策略

跨 marker 遷移的差距主要出現在 **N < 10** 的區間；N 很大時兩條曲線會收斂。

**這決定的是取樣位置，不是計劃價值**——曲線必須在低 N 密集取樣（1, 2, 3, 5, 8），高 N 稀疏即可（20, 50）。若只在高 N 取樣，會量到一個貼合的收斂區段而錯過整個效果。

價值主張是「**用少很多的標註達到可用水準**」，不是「不用標註」。這個期待值最好在實驗開始前就建立共識。

**選哪個 marker 當實驗平台**：挑**標註最豐富**的那一個。可以對它子取樣成任意 N，同時仍保有夠大的保留測試集讓誤差條收斂。標註稀少的 marker 做不出可信的曲線——測試集太小，雜訊會蓋過訊號。

### 一個會讓實驗失效的陷阱

若來源與目標 marker 的**取像條件也不同**（例如一個 z = 4 µm、另一個 z = 2 µm），遷移失敗時將無法區分是因為換了 marker，還是因為物件在模型眼中的尺度差了兩倍。

**spacing 對齊必須在遷移實驗之前完成**（6.1 節）。這是資料契約留在高優先序的主要理由。

### 附帶實驗

同一批設定下，額外跑一組「截斷 encoder 至 stride 8」的曲線，驗證 2.3 節的深度取捨。成本很低，但能把架構決策從推論變成量測。

---

## 8. 增強與域隨機化

在本計劃中屬次要（若來源與目標 marker 使用同一套取像流程，跨域不是主要矛盾），但仍影響遷移後的穩健性，成本極低。

**現況**（`train.py` 頂部的模組層級常數）：

```python
RandFlipd(spatial_axis=1, prob=0.5),        # 只翻一個軸
RandAdjustContrastd(prob=0.3),
# RandGaussianNoised(prob=0.4, ...),         ← 被註解掉
RandBiasFieldd(prob=0.2),
RandShiftIntensityd(offsets=0.2, prob=0.3),
RandScaleIntensityd(factors=0.2, prob=0.3),
```

三個缺口：**雜訊增強被註解掉**（模型沒見過訊噪比變化）、**只翻一個軸**、**完全沒有模糊增強**。

另有一個架構不一致：整個專案是 config 驅動（正規化、模型、損失、指標都從設定檔建立），**唯獨 transform 寫死在 `train.py` 頂部**。這使增強策略無法掃參、無法記錄進 artifacts、無法針對不同 marker 使用不同配方。

**修改方向**：移入 `utils/transforms.py` 並改為 config 驅動，補上三軸翻轉與旋轉、取消雜訊增強的註解並提高強度、加入各向異性隨機模糊與模擬低 z 解析度。

⚠️ 加強增強後 in-domain 驗證損失會變差，這是預期行為。判斷有效與否一律看保留集的 detection 指標。

---

## 9. PINN 與 PSF re-engineering

屬探索性項目，優先序低於前八節，但記錄完整脈絡供後續參考。

### 9.1　名詞釐清（一）：兩種 PINN

**嚴格定義的 PINN**（Raissi 那一脈）是把偏微分方程的殘差寫進損失，讓網路輸出滿足該方程。**在顯微去卷積上，這種 PINN 幾乎不存在**，原因很單純：

$$I = \text{PSF} \otimes o + \varepsilon$$

螢光成像的正向模型是**卷積／積分方程，不是偏微分方程**。沒有微分方程殘差可罰。（真正會出現 PDE 的是光在樣本中傳播那一層，但那是模擬光學系統，不是影像復原。）

**廣義的 physics-informed**——把成像物理當成網路結構或損失的一部分——這個非常多，而且是主流。文獻上常自稱 PINN，實質是 *model-based deep learning*。

**方向對，關鍵字要換**：

> `algorithm unrolling`、`physics-informed image restoration`、
> `differentiable forward model`、`blind deconvolution with parameterized PSF`

若照「PINN」搜尋，會撈到一堆流體與熱傳的論文，然後誤以為這個領域沒人做。

### 9.2　名詞釐清（二）：兩種 PSF re-engineer

**(A) 計算上重新估計 PSF**——從影像本身反推（盲／半盲去卷積），不需重拍。**對既有資料可行。**

**(B) 物理上重新設計 PSF**——在傅立葉平面加相位遮罩，把 PSF 改成 Tetrapod、double-helix 之類的形狀，讓深度資訊編碼進強度圖案。代表作 DeepSTORM3D（Nature Methods 2020）。**需改動光路硬體並重新拍攝，對既有資料不適用。**

⚠️ **待查證**：從推論路徑中的 `Destripe` 判斷，資料可能來自光片顯微鏡。光片的軸向 PSF 主要由**光片厚度**決定，而非物鏡 NA。以 Zernike 多項式參數化物鏡像差的做法是為共軛焦／寬場設計的，**對光片系統的適用性尚未確認**，實作前需先釐清成像模態。

### 9.3　文獻地圖

**Algorithm unrolling —— 最相關**

**RLN（Richardson–Lucy Network, Nature Methods 2022, Li 與 Shroff 團隊）** 把 RL 迭代的正向／反向投影結構直接寫成網路層。報告重點正是泛化性：

- 僅含約 **16,000 個參數**，比純資料驅動網路快 4 到 50 倍
- 更好的去卷積、**更好的泛化性**、更少的假影，軸向尤其明顯
- 明確做了泛化測試（一種樣本訓練、測另一種），比純資料驅動結構產生更少假影
- 參數量不到 CARE 與 RCAN 的 1/60
- 即使**只用合成資料訓練**也優於傳統 RL
- 在寬場、光片、共軛焦、超解析多模態上展示

核心啟示：把物理顯式寫進網路結構，可學習自由度從幾百萬壓到一萬多，泛化性反而上去。這與 6.3 節「依物件尺寸決定網路深度、增加歸納偏置」是同一個機制。

**明確自稱 PINN 的工作**

**m-rBCR**（ECCV 2024）借用 Beylkin–Coifman–Rokhlin 的非標準型壓縮方案，提出 physics-informed 的 Multi-Stage Residual-BCR Net。動機直白：傳統方法依賴取像時的 PSF，但常因 **PSF 模型不準與雜訊假影**而失效。結果在兩個真實顯微資料集與模擬 BioSR 上勝過 RL、U-Net、DDPM、CARE、DnCNN、RCAN 等，**參數量少 30 到 210 倍，執行快 3 到 300 倍**。

**不需標準答案的物理先驗方法**

- **DPS**：把成像過程、樣本先驗與 deep back-projection 嵌入去卷積策略，**不需高品質 ground truth** 即達成約 1.67 倍解析度提升
- **自監督 PSF-informed 去卷積**（Advanced Imaging 2025，OCT 領域，方法可移植）：結合去雜訊、盲 PSF 估計與稀疏去卷積，**只用含雜訊的原始掃描**
- **PI-AstroDeconv**：把 PSF 當先驗整合進 encoder–decoder，**可處理多個 PSF、能容忍量測不精確、不依賴標註**——「容忍不精確的 PSF」正是實務上最需要的容錯性
- **Zero-shot / Deep Image Prior**：單張影像最佳化，但自承限制——去卷積是病態反問題且對雜訊極敏感，**低訊噪比下常放大假影**

### 9.4　與本計劃的關係

**相同的哲學**：能用物理解析地解釋掉的變異，就不要叫網路去學。

**必須講清楚的差異**：

1. **任務性質不同。** 去卷積是 restoration（有正向模型可驗證重建誤差）；分割是 discrete labeling（沒有正向模型能從標籤生成觀測）。損失不能直接搬用。

2. **PSF 與本計劃核心目標的關聯有限。** 本計劃要解的是跨 marker 遷移。PSF 屬於取像條件，只有在來源與目標的取像方式也不同時才成為因素。**若使用同一套取像流程，PSF 不是主要矛盾。**

3. **去卷積可能反而害了下游。** 病態反問題會放大雜訊、製造 ringing 假影，在低訊噪比影像上有機會**憑空生出假的物件**。驗收必須用 detection F1 與計數誤差，**不能用 PSNR / SSIM 交差**。

### 9.5　三個階梯（條件性執行）

**階梯一（零成本，與第 8 節合併）**：在增強中加入各向異性隨機模糊與模擬低 z 解析度。這是 PSF 域隨機化的最低成本版本。**價值在於便宜地否證**——若無效，代表瓶頸不在光學，不應投入階梯二、三。

**階梯二（低成本，1 個月）**：拍 sub-resolution beads 量測實測 PSF，用 RL 或 RLN 去卷積到共同的目標 PSF 當作前處理。可直接量化 PSF 佔問題的比例。前提是顯微鏡仍可取用。

**階梯三（研究級）**：可微分的參數化 PSF 前置層，骨幹凍結，以自監督重建損失擬合十幾個係數，**不需新標註**。參數具物理意義、可與 bead 實測交叉驗證。僅在階梯一或二顯示明確效益時執行。

---

## 10. 路線圖

| 階段 | 週次 | 內容 | 交付物 |
|---|---|---|---|
| **P0**　量測可信化 | 1 | `analysis3d.py`（先單獨跑）、seed 控制、animal-level 切分、錯誤日誌 | 診斷表：2D vs 3D 計數差距、洩漏幅度 |
| **P1**　遷移基礎建設 | 2–3 | checkpoint 規格（含血統欄位）、載入與分段解凍、學習率分組、InstanceNorm、舊權重轉檔 | **可運作的微調流程** |
| **P2**　資料契約 | 3–4 | 百分位正規化、spacing 對齊、fingerprinting、標註來源檢測 | 可重用的前處理契約 |
| **P3**　hub v0 | 4–5 | 取 Cellpose 或自監督預訓練當 `shared_init v0`，跑通 hub → marker 的完整流程 | **流程打通**（內容尚未最佳化） |
| **P4**　核心實驗 | 6–7 | 標註效率曲線（4 初始化 × 6 標註量 × 3 seed）+ encoder 截斷附帶實驗 | **本計劃的主要成果** |
| **P5**　hub v1 | 8–9 | 多 marker 共同訓練，blob / tubular 兩個 head，與 v0 對比驗證 | 版本化的 `shared_init v1` |
| **P6**　任務適配 | 10+ | task_family 分流、輸出頭、依 fingerprint 決定截斷深度 | 每個 family 的最佳配置 |
| **P7**　PSF（條件性） | — | 階梯一 → 視結果決定是否往下 | 論文級成果或明確否證 |

**P3 與 P5 分開是刻意的**：先把 hub 這個流程跑起來（即使內容只是現成權重），比先把 hub 內容做到最好重要。

### 若時間有限，按此順序

1. checkpoint 規格 + 微調路徑（這就是要的功能本身）
2. animal-level 切分 + 3D detection 指標（沒有這個，實驗做完讀不出結論）
3. 隨機性控制（各組 arm 要能公平比較）
4. 百分位正規化 + spacing 對齊（checkpoint 可重用的前提）
5. hub v0 跑通流程
6. 標註效率曲線
7. dataset fingerprinting
8. 標註來源檢測
9. hub v1（多 marker 共同訓練）
10. 增強強化
11. task family 分流與輸出頭
12. PSF 探索

---

## 11. 驗收指標

### 主要

| 指標 | 定義 |
|---|---|
| **標註效率比** | 達到目標 detection F1 所需標註量，隨機初始化 ÷ hub 初始化 |
| **detection F1** | 3D 質心配對，容差取物件半徑（由 fingerprint 決定） |
| **計數 MAPE** | 每 volume 的物件數相對誤差 |
| **clDice**（管狀類） | 拓樸保真度 |

**參考指標**：voxel Dice、IoU。僅供對照，**不作為選模型或報告依據**——對僅數個體素寬的物件，單一體素誤差就改變 Dice 約 10%。

### hub 版本驗收

新版 `shared_init` 發布前，必須在**至少兩個不同 marker** 上跑標註效率曲線，且皆不劣於前一版，才可切換預設。

### 最終科學驗收

更換模型後，處理組與對照組的差異是否仍能複現？這比任何 Dice 都重要。

⚠️ **偏差警訊**：若模型對暗淡物件的偵測率較低，而某一實驗組整體較暗，模型會**憑空製造出組間差異**。務必在所有實驗組上均等抽樣做人工驗證，不可只驗證看起來漂亮的那組。

---

## 12. 待決事項

| # | 事項 | 影響 |
|---|---|---|
| 1 | 哪個 marker 的標註**物件數**最多、涵蓋幾隻動物？ | 決定拿哪個 marker 當第 7 節的實驗平台。標註越豐富越適合——可子取樣成任意 N，同時保有夠大的測試集。（注意單位是標註物件數與動物數，不是 crop 檔案數） |
| 2 | `training-data` 底下每個 marker 各有幾個 volume、幾隻動物？ | 決定 animal-level 切分是否可行，以及交叉驗證折數 |
| 3 | 各 marker 的取像條件是否相同？ | 若不同，需先做 spacing 對齊，否則第 7 節實驗無法歸因 |
| 4 | 磁碟餘裕？ | 決定 spacing 目標採方案 ① 或 ③ |
| 5 | 成像模態確認（光片？共軛焦？） | 決定 9.2 節 PSF 路線的適用性 |
| 6 | 現有標註是逐物件 instance 還是單一二值遮罩？ | 決定質心輸出頭能否直接從現有標註生成 |
| 7 | 各 marker 的典型物件直徑（µm）？ | 可由 6.3 的 fingerprinting 自動量測，但初期需人工確認一次以驗證量測正確 |

---

*本文件基於對 repository 全部 33 個原始碼檔案的逐檔檢查撰寫。所引用的程式碼行為皆已實際確認。文獻部分數據引自各論文摘要與結果段落，正式引用前建議核對原文。標記為「待查證」之處需由團隊確認後才能定案。*
