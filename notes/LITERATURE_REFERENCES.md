# 文獻參考清單

依改造計劃書的章節組織。每則標註：**支持哪一節**、**可引用的具體內容**、**怎麼用**。

---

## ★ 最重要的一篇：幾乎是你們計劃的完整先例

### Scheinfeld et al. (2026) — A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring
`arXiv:2605.26026`　Weill Cornell / Helmholtz Munich（Paetzold、Ertürk 等人）
公開權重與程式碼：`https://github.com/AdinaScheinfeld/lsm_fm_public_repo.git`

**支持**：1.1 hub 設計、4.3 分段解凍、5.1 初始化來源、6.2 正規化、7 標註效率曲線、2.4 形態分類

這篇把你們要做的事整套做完了，而且用的是同一套技術棧。重點對照：

| 你的計劃 | 該論文的做法 |
|---|---|
| shared_init hub | 3D 基礎模型，跨物種／染色／取像協定預訓練 |
| 分段解凍（4.3） | **encoder 凍結 warmup → 以較低學習率解凍；decoder 全程用完整學習率** |
| 百分位正規化（6.2） | 預訓練與微調**一律用百分位強度重標到 [0,1]** |
| 域隨機化增強（8） | 翻轉、旋轉、仿射、高斯雜訊、**高斯平滑**、強度縮放與平移 |
| blob / tubular 分家（2.4） | 三種結構：amyloid plaque（稀疏點狀）、cell nucleus（密集孤立）、vessel（連續管狀） |
| detection 指標（11） | 同時報 **total Dice 與 instance Dice** |
| 標註效率曲線（7） | few-shot = 5 patches、many-shot = 15 patches，3-fold CV |
| 模型骨幹 | **MONAI 的 UNet 與 SwinUNETR**——跟你們 repo 相同 |

**最值得引用的數字**（Table 1，few-shot = 5 個 patch）：

- **Amyloid plaque**（稀疏點狀，最接近 c-Fos 的情境）：從零訓練的 UNet 是 Tot 0.50 / **Inst 0.11**；預訓練後是 Tot 0.68 / **Inst 0.58**。instance Dice 從 0.11 拉到 0.58。
- **Cell nucleus**：Cellpose 3D 是很強的基線（0.79 / 0.80）；µSAM 在這個任務上 **instance Dice 只有 0.00–0.12**——2D 基礎模型不保證能轉到 3D。
- **Vessels**：預訓練 0.87–0.89，優於從零的 0.83–0.86。

**怎麼用**：
1. **直接當 `shared_init v0` 的候選**——權重是公開的，且用的就是 MONAI UNet / SwinUNETR，載入相對容易。這可能讓 P3 從「幾週」變成「幾天」。
2. 計劃書第 7 節的實驗設計可以直接對齊他們的（5 / 15 patch、3-fold CV），這樣結果可以互相比較。
3. 引用它來支持「預訓練確實降低標註需求」——而且他們還做了 6 位專家的盲測評分，不只是指標好看。

⚠️ 這是 2026 年 5 月的 preprint，尚未經同儕審查，引用時要註明。

---

## 資料集與挑戰賽

### SELMA3D Challenge — Self-supervised learning for 3D light-sheet microscopy image segmentation
`arXiv:2501.03880`；`selma3d.grand-challenge.org`（2024 / 2025 / 2026 三屆）

**支持**：2.4 形態分類、5.1 自監督預訓練、5.2 版本策略

**兩個關鍵價值**：

**(1) 公開的預訓練資料集。** 2024 年版提供 35 張大型 3D 影像（每張超過 1000³ 體素）加上 315 個標註 patch；2025 年版擴充到 58 張影像，並從只有腦部擴展到全身多種器官與結構。**這是你們自監督預訓練可以直接用的資料**，不必只靠自己的無標註資料。

**(2) 官方採用的結構二分法，與你的 blob / tubular 完全一致。** 他們的用詞是：
- **isolated structures**（孤立結構）——細胞核、**c-Fos 陽性細胞**、amyloid 斑塊、微膠細胞
- **contiguous structures**（連續結構）——血管、神經

2025 年起正式拆成兩個 task。這是你 2.4 節設計的外部佐證，而且**術語可以直接沿用**，讓你們的文件跟社群對齊。

該挑戰賽的動機陳述也可直接引用：LSM 領域存在大量不同生物結構的未標註資料，但自監督學習在此領域尚未被充分探索；因此有必要發展能服務多種 LSM 分割任務的通用／基礎模型。

---

## 基礎模型與遷移起點（對應 5.1 節）

### Pachitariu, Rariden & Stringer (2025) — Cellpose-SAM: superhuman generalization for cellular segmentation
`bioRxiv 2025.04.28.651001`

**支持**：5.1 初始化來源 C、8 域隨機化、11 標註者一致性

把 SAM 的預訓練 transformer 骨幹接進 Cellpose 框架。**結果顯著超越標註者間一致性，逼近「人類共識」上限**。

**兩點對你特別有用**：

1. **他們明確論證「標註者間一致性不是效能上限」**——理論上的共識標註可以把錯誤率再減半。這修正了我先前把 inter-annotator F1 當天花板的說法，值得在你的驗收指標一節註明。

2. **他們提升泛化性的手段，跟計劃書第 8 節的清單高度重疊**：對通道洗牌、細胞大小、shot noise、降採樣、**等向與非等向模糊**都做了穩健化。這是「各向異性模糊增強」這項建議的直接文獻支持。

模型可直接接入 Cellpose 生態系（微調、human-in-the-loop、影像復原、3D 分割）。

相關：
- Stringer & Pachitariu, **Cellpose3**: one-click image restoration for improved cellular segmentation. *Nat Methods* 22, 592–599 (2025)
- Stringer et al., **Cellpose** 1.0. *Nat Methods* 18, 100–106 (2021)
- Pachitariu & Stringer, **Cellpose 2.0**: how to train your own model. *Nat Methods* (2022)

### Archit et al. (2025) — Segment Anything for Microscopy (µSAM)
*Nature Methods* 22, 579–591　`github.com/computational-cell-analytics/micro-sam`

**支持**：4.1 分層遷移、5.1 來源 C、2.2 head 分家

**最相關的技術細節**：他們在 SAM 上**加了一個新的 decoder，同時預測前景、以及到物件中心與邊界的距離**，再用後處理得到自動 instance 分割。

這正是計劃書 2.4 節「blob head = 前景 + 質心熱圖」的同構設計，而且是已發表的做法。可以直接引用來支持這個架構決策。

另外他們用 **LoRA（rank 4）** 做資源受限情境下的微調，可作為 4.5 節「輕量適應」的參考。

⚠️ 但要注意上面 LSM 基礎模型論文的結果：**µSAM 在 3D 細胞核任務上的 instance Dice 只有 0.00–0.12**。2D 基礎模型不保證能轉到 3D 體積資料。這是「不要預設任何來源一定可行」的實證。

### Marks et al. (2025) — CellSAM: A Foundation Model for Cell Segmentation
*Nature Methods* 22, 2585–2593

**支持**：5.1 來源 C

以 SAM 為基礎，發展 prompt engineering 方式產生遮罩。報告指出在幾乎所有比較中 CellSAM 都優於 generalist Cellpose 模型，並與各資料集專門訓練的 specialist Cellpose 相當。

論文中對「把 SAM 從自然影像轉到細胞影像」的困難描述值得參考：效能下降、邊界不確定，且細胞影像有額外複雜度——不同成像模態、單一視野內可能有數千個物件（自然影像通常只有數十個）。

### Achard et al. (2025) — CellSeg3D: self-supervised 3D cell segmentation for microscopy
*eLife* 13, RP99848

**支持**：5.1 來源 D

3D 自監督細胞分割，可作為基線或起點候選。

---

## 架構選擇（對應 2.3、6.3 節）

### Isensee, Wald, Ulrich et al. (2024) — nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation
MICCAI 2024, LNCS 15009, pp. 488–498　`arXiv:2404.09556`

**支持**：2.3 encoder 深度、3.3 切分、11 驗收

**核心結論可直接引用**：許多宣稱勝過 U-Net 基線的新架構，在檢視常見的驗證缺陷（基線設定不足、資料集不足、算力資源未對齊）後都站不住腳。嚴格驗證下，達到 SOTA 的配方是：(1) 使用 CNN-based U-Net 模型，包含 ResNet 與 ConvNeXt 變體；(2) 使用 nnU-Net 框架；(3) 將模型規模擴展到現代硬體。

比較對象**明確包含 Transformer-based 與 Mamba-based 方法**。他們也提出了 Residual Encoder U-Net 的標準化基線（M / L / XL 三種硬體規模）。

**怎麼用**：
1. 支持「把 SwinUNETR 降為對照組、以 ResEnc U-Net 為主力」的建議
2. 更重要的是**支持整份計劃書的方法論立場**——他們指出領域存在「偏好新架構的創新偏誤」，需要更嚴格的驗證標準。你的 P0 階段做的正是這件事
3. 他們還提出一個「衡量資料集是否適合做方法比較」的策略，對你選擇實驗平台的 marker 有參考價值

---

## 評估與資料洩漏（對應 3.3、3.4、11 節）

### Maier-Hein, Reinke, Godau et al. (2024) — Metrics reloaded: recommendations for image analysis validation
*Nature Methods* 21(2), 195–212　線上工具：Metrics Reloaded online tool

**支持**：11 驗收指標、3.4 評估、6.3 fingerprinting

國際共識框架（Delphi 流程），指導如何為特定問題選對指標。核心概念是 **problem fingerprint**——把問題的各面向（領域關切、目標結構性質、資料集、演算法輸出）結構化表示，據此選擇指標並提示常見陷阱。

涵蓋 image-level classification、**object detection**、semantic segmentation、**instance segmentation** 四類。

**怎麼用**：
1. **這是「不要只報 Dice」最權威的引用來源**。他們明確指出：所選的效能指標常無法反映領域關切，因而無法真正衡量科學進展
2. 你的任務在他們的分類下屬於 **instance segmentation / object detection**，不是 semantic segmentation——這個界定本身就支持改用 detection F1
3. 有配套的線上工具可以照著跑一遍，產出的建議可以直接寫進方法章節

配套文章：Reinke et al., **Understanding metric-related pitfalls in image analysis validation**, *Nat Methods* (2024)。

### 資料洩漏的量化證據（支持 3.3 節）

這幾篇提供**具體數字**，比抽象論述有說服力得多：

**Yagis et al.**（經 MDPI 2025 的範疇綜述引述）：同一個資料集，**slice-wise 切分得到 94% 準確率，subject-wise 切分只有 66%**——28 個百分點的落差**完全來自驗證協定，與模型能力無關**。

> 這是你跟老師解釋「數字會下降」時最好的一句引用。

**Bussola et al.**：數位病理領域，當同一病人的 tile 同時出現在訓練與驗證集時，**效能估計可被灌水達 41%**。

**Yagis et al. (2021)**, *Scientific Reports* — Effect of data leakage in brain MRI classification using 2D CNNs：另有一個反直覺的發現——使用 slice-level 切分時，**在 34 個受試者的小資料集上得到的分數竟高於 200 個受試者的資料集**。他們的結論是：**資料洩漏在小資料集時影響尤其嚴重**。

> 這一點對你特別關鍵。標註效率曲線的低 N 區間正是小資料集，如果不修 3.3，那個區間的曲線會被洩漏完全污染。

**Tampu et al. (2022)**, *Scientific Data* — Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images：OCT 的情境跟你們幾乎一樣——由於系統的微米級解析度，**相鄰切片在結構與雜訊上都非常相似**，不當切分導致訓練與測試集重疊。量化結果：MCC 灌水 0.07 到 0.43（準確率 5% 到 30%）。

> 「相鄰切片高度相似」這個論證可以直接移植到你們的 patch-level 切分。

**Rumala (2023)**, `arXiv:2309.00350` — How You Split Matters：用 GradCAM 顯示 CNN 學到的捷徑來自 **identity confounding**——模型學會辨識個體，而非診斷特徵。

---

## 管狀結構（對應 2.4 節）

### Shit et al. (2021) — clDice: a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation
`arXiv:2003.07311`（CVPR 2021）

**支持**：2.4 tubular family 的損失與指標

血管／管狀結構的拓樸感知損失。你們 repo 的 metrics registry 裡已經有，但目前沒有跟 task_family 綁定。

### Todorov, Paetzold, Schoppe et al. (2020) — Machine learning analysis of whole mouse brain vasculature
*Nature Methods* 17(4), 442–449

**支持**：2.4 tubular、下游科學驗證

全鼠腦血管的機器學習分析，是 LSM 血管分割的標竿工作。與你們的 Lectin / CD31 任務直接相關。

### Weigert et al. (2020) — Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy (StarDist-3D)
WACV 2020, pp. 3666–3673

**支持**：2.4 blob family、5.1 來源

3D 星凸多面體表徵，對密集排列的細胞核 instance 分割特別有效。是 blob head 的另一個設計選項。

---

## 物理啟發的影像復原（對應第 9 節）

### Li, Su, Guo et al. (2022) — Incorporating the image formation process into deep learning improves network performance
*Nature Methods* 19, 1427–1437（RLN）　`github.com/eguomin/regDeconProject`

**支持**：9.3 algorithm unrolling、以及「參數少 → 泛化好」的一般性論證

已核對過的具體數據：

- RLN 把傳統 Richardson–Lucy 迭代與全卷積網路結構結合，**建立與影像形成過程的連結**
- **僅含約 16,000 個參數**
- 比參數量大得多的純資料驅動網路**快 4 到 50 倍**
- 提供更好的去卷積、**更好的泛化性**、更少的假影，**軸向尤其明顯**
- 參數量**不到 CARE 與 RCAN 的 1/60**
- 在嚴重離焦螢光或雜訊污染的體積上勝過傳統 RLD
- **即使只用合成資料訓練**，軸向解析度仍優於 RLD，低訊噪比下亦然
- 在清透組織大型資料集上，比傳統 multi-view pipeline 快 4 到 6 倍
- 展示涵蓋寬場、光片、共軛焦、結構照明顯微鏡

論文中對 algorithm unrolling 的定義值得引用：用網路層表示傳統迭代演算法的每一步（如 ADMM-net、ISTA-net），資料通過展開後的網路，等同於執行有限次數的迭代。

> **這是第 9 節「關鍵字不是 PINN 而是 algorithm unrolling」的直接依據，也是「顯式寫入物理 → 參數大幅減少 → 泛化更好」這個論證的最強證據。**

相關後續：**LUCYD**（MICCAI 2023, Chobola et al.）——同樣結合 RL 公式與深度特徵，並以 RLN 為主要基線。

### 其他 physics-informed 去卷積

- **m-rBCR**（ECCV 2024, Li, Kudryashev & Yakimovich）——自稱 physics-informed，參數少 30–210 倍
- **DPS**（deep-physics-informed sparsity）——不需高品質 ground truth
- **PI-AstroDeconv**——可容忍不精確的 PSF 量測、不依賴標註

⚠️ 這三則我先前是從摘要與結果段落取得數據，**正式引用前建議自行核對原文**。

### PSF engineering（硬體那條路，供背景參考）

- **DeepSTORM3D**（Nehme et al., *Nature Methods* 2020）——Tetrapod PSF 的密集定位與 PSF 設計

⚠️ 需改動光路，對既有資料不適用。且**光片系統的軸向 PSF 主要由光片厚度決定而非物鏡 NA**，Zernike 像差參數化的適用性需先確認成像模態。

---

## LSM 分析的應用端參考

- **Kaltenecker et al. (2024)** — Virtual reality-empowered deep-learning analysis of brain cells. *Nature Methods* 21(7), 1306–1315
- **Mai et al. (2024)** — Whole-body cellular mapping in mouse using standard IgG antibodies. *Nature Biotechnology* 42(4), 617–627
- **Zhao et al. (2020)** — Cellular and Molecular Probing of Intact Human Organs. *Cell* 180(4), 796–812
- **Voigt et al. (2019)** — The mesoSPIM initiative: open-source light-sheet microscopes for imaging cleared tissue. *Nature Methods* 16(11), 1105–1108
- **Stelzer et al. (2021)** — Light sheet fluorescence microscopy. *Nature Reviews Methods Primers* 1, 73（綜述）
- **Ma et al. (2024)** — The multimodality cell segmentation challenge: toward universal solutions. *Nature Methods* 21, 1103–1113

---

## 依計劃書章節的對應索引

| 計劃書章節 | 主要文獻 |
|---|---|
| 1.1 hub 設計 | Scheinfeld 2026；SELMA3D |
| 2.2 encoder 共用 / head 分家 | µSAM（前景 + 中心距離 decoder）；Scheinfeld 2026 |
| 2.3 encoder 深度 | nnU-Net Revisited |
| 2.4 blob / tubular | SELMA3D（isolated / contiguous 官方術語）；clDice；Todorov 2020；StarDist-3D |
| 3.3 patch-level 切分 | Yagis（94%→66%）；Bussola（41%）；Tampu OCT；Rumala |
| 3.4 3D 評估 | Metrics Reloaded |
| 4.3 分段解凍 | Scheinfeld 2026（凍結 warmup → 低 LR 解凍） |
| 5.1 初始化來源 | Cellpose-SAM；µSAM；CellSAM；CellSeg3D；Scheinfeld 2026 |
| 6.2 百分位正規化 | Scheinfeld 2026 |
| 7 標註效率曲線 | Scheinfeld 2026（5 / 15 patch、3-fold CV） |
| 8 域隨機化 | Cellpose-SAM（非等向模糊等）；Scheinfeld 2026 |
| 9 PINN / PSF | RLN；LUCYD；m-rBCR；DeepSTORM3D |
| 11 驗收指標 | Metrics Reloaded；Cellpose-SAM（一致性非上限） |

---

## 三個需要修正計劃書的地方

**(1) 「標註者一致性是效能天花板」需要限縮。**
Cellpose-SAM 明確論證這不是真正的上限——理論上的共識標註可把錯誤率減半，而他們的模型已超越標註者間一致性。計劃書 6.4 節與驗收章節應改為「一致性是**參考水準**」而非「物理上限」。

**(2) blob / tubular 應改用社群術語。**
SELMA3D 用的是 **isolated structures** 與 **contiguous structures**，且已是挑戰賽的正式分類。改用這組詞可讓文件與社群對齊，也方便引用。

**(3) `shared_init v0` 的候選應該增加一項。**
Scheinfeld 2026 的 LSM 基礎模型權重是公開的，而且骨幹就是 MONAI UNet / SwinUNETR——跟你們 repo 相同。這比 Cellpose（2D 為主）更貼近你們的資料型態，應該列為 v0 的第一順位候選。

---

*本清單中的引用資訊皆經檢索核對。標記 ⚠️ 者為僅根據摘要與結果段落取得，正式引用前請核對原文。arXiv:2605.26026 為 2026 年 5 月的 preprint，尚未經同儕審查。*
