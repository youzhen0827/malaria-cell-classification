# 瘧疾細胞影像分類 🔬

機器學習課程期末報告，使用 TensorFlow Datasets 的 Malaria 瘧疾細胞顯微鏡影像資料集，比較三種卷積神經網路架構在二元分類任務上的效能差異。

🔗 **[程式碼（Google Colab）](https://colab.research.google.com/drive/1H_tkPPHeYq-oCneCc8B9Pd_l2xq8Ug7i?usp=sharing)**

---

## 研究背景

瘧疾診斷通常依賴人工鏡檢，過程耗時且易出錯。本研究透過深度學習自動辨識細胞薄片顯微鏡圖像，輔助自動化篩檢系統，判斷細胞是否遭瘧原蟲寄生。

---

## 資料集

- **來源**：TensorFlow Datasets（TFDS）malaria 資料集
- **總數**：27,558 張細胞薄片顯微鏡圖像
- **類別**：二元分類
  - `Parasitized`：被寄生（含瘧原蟲）
  - `Uninfected`：未感染

---

## 模型對比

| 模型 | 架構 | 準確率 | 特點 |
|------|------|--------|------|
| **Model A**（基礎 CNN） | 2 層卷積，無 BN / Dropout | ~82–85% | 架構簡單、訓練快，但過擬合嚴重 |
| **Model B**（自創優化） | 4 層卷積 + BN + Dropout(0.5) | **~94–96%（最高）** | 特徵捕捉精準，表現最穩定 |
| **Model C**（MobileNetV2） | Google 預訓練遷移學習 | ~90–92% | 收斂最快，但針對性稍弱 |

**最終選定 Model B 為最佳模型**，在保持運算效率的同時，透過深層卷積與防過擬合機制達到最高預測可靠度。

---

## Model B 設計細節

針對瘧疾細胞微小病徵（瘧原蟲紫色小點）進行四項優化：

**1. 深層特徵提取**
- 卷積層由 2 層增加至 4 層
- 強化對微小病徵的捕捉能力

**2. 訓練穩定化**
- 每個卷積層後加入 Batch Normalization
- 克服梯度消失問題，加速收斂

**3. 強化泛化能力**
- 全連接層配置 Dropout(0.5)
- 防止過擬合，提升泛化能力

**4. 提升分類容量**
- 分類層神經元提升至 512 個
- 應對醫學影像中細微的分類邊界

---

## 技術棧

```
Python
TensorFlow / Keras
TensorFlow Datasets (TFDS)
MobileNetV2（遷移學習）
Google Colab
```

---

## 學習成果

- 比較 From Scratch CNN 與 Transfer Learning 在醫療影像上的效能差異
- 理解 Batch Normalization 對深層網路訓練穩定性的影響
- 透過 Dropout 解決過擬合問題
- 驗證針對性設計的自創架構在特定領域（醫療影像）優於通用預訓練模型的場景
