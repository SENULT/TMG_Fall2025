# 📚 Hướng Dẫn Preprocessing cho Phân Loại Cảm Xúc Tiếng Việt

## TMG Fall 2025 - Vietnamese Sentiment Classification

---

## 🎯 Mục Tiêu

Xây dựng pipeline preprocessing hoàn chỉnh cho bài toán **phân loại cảm xúc (sentiment classification)** và **phân loại chủ đề (topic classification)** trên văn bản tiếng Việt.

### Bài Toán
- **Input:** Văn bản tiếng Việt (comments, reviews, posts)
- **Output:** 
  - **Sentiment:** Cảm xúc (0.0=Neutral, 1.0=Positive, 2.0=Negative, 3.0=Mixed)
  - **Classification:** Chủ đề/Topic (0.0-9.0, tổng 10 categories)

---

## 📊 Dataset Overview

### Thống Kê Dataset

| Split | Samples | Ratio | Avg Length | Avg Words |
|-------|---------|-------|------------|-----------|
| **Train** | 23,050 | 70% | 105 chars | 25 words |
| **Val** | 4,940 | 15% | 105 chars | 25 words |
| **Test** | 4,940 | 15% | 104 chars | 25 words |
| **Total** | **32,930** | 100% | 105 chars | 25 words |

### Phân Bố Sentiment

**Train Set:**
- Sentiment 0.0 (Neutral): **15,936** (69.1%) ✅ Dominant
- Sentiment 2.0 (Negative): **3,634** (15.8%)
- Sentiment 1.0 (Positive): **2,901** (12.6%)
- Sentiment 3.0 (Mixed): **579** (2.5%) ⚠️ Minor

**Đặc điểm:**
- Dataset **không cân bằng** (imbalanced)
- Class Neutral chiếm ưu thế (~69%)
- Class Mixed rất nhỏ (<3%) → Cần chú ý khi training

### Phân Bố Topic/Classification

**Top 5 Topics (Train Set):**
1. Topic 3.0: **10,071** (43.7%) - Dominant
2. Topic 2.0: **7,356** (31.9%)
3. Topic 4.0: **1,650** (7.2%)
4. Topic 6.0: **1,034** (4.5%) ⚠️ Minor
5. Topic 1.0: **626** (2.7%) ⚠️ Minor

**Topics ít mẫu (< 5%):**
- Topic 6.0, 1.0, 5.0, 7.0, 8.0, 9.0, 0.0

---

## 🔧 Pipeline Preprocessing

### Tổng Quan Flow

```
Raw CSV Data (32,966 samples)
    ↓
[1. Load & Combine]
    ↓
[2. Text Normalization]
    ↓
[3. Special Token Mapping]
    ↓
[4. Teencode Handling]
    ↓
[5. Sentence Splitting]
    ↓
[6. Length Filtering] (→ 32,944 samples, -22)
    ↓
[7. Deduplication] (→ 32,930 samples, -14)
    ↓
[8. Remove Small Classes] (stratification safe)
    ↓
[9. Stratified Split] (70/15/15)
    ↓
Train/Val/Test Parquet Files
```

---

## 📝 Chi Tiết Từng Bước

### 1️⃣ Text Normalization (Chuẩn Hoá Văn Bản)

**Mục đích:** Đưa văn bản về dạng chuẩn, nhất quán

#### a) Lowercase
```python
# Before: "Xin CHÀO Mọi Người!"
# After:  "xin chào mọi người!"
text = text.lower()
```

**Lợi ích:**
- Giảm vocabulary size
- Model không phân biệt "CHÀO" vs "chào"
- Tăng khả năng generalization

#### b) Giữ Dấu Tiếng Việt
```python
# QUAN TRỌNG: Không bỏ dấu!
# Giữ nguyên: "hòa" ≠ "hoà" ≠ "hoa"
keep_vietnamese_tones: true
```

**Tại sao?**
- Tiếng Việt là **ngôn ngữ thanh điệu**
- Dấu thay đổi nghĩa: "ma" (ghost) vs "má" (mother) vs "mà" (but)
- Bỏ dấu → mất thông tin quan trọng cho sentiment

#### c) Chuẩn Hoá Khoảng Trắng
```python
# Before: "xin   chào     bạn"
# After:  "xin chào bạn"
text = re.sub(r'\s+', ' ', text).strip()
```

---

### 2️⃣ Punctuation Normalization (Chuẩn Hoá Dấu Câu)

**Vấn đề:** User thường dùng nhiều dấu để thể hiện cảm xúc

```python
# Before: "Tuyệt vời quá!!!!!!!"
# After:  "Tuyệt vời quá!!!"  (giới hạn 3)

# Before: "Không biết............"
# After:  "Không biết..."

max_consecutive_punct = 3
```

**Lý do giới hạn:**
- `!!!` vs `!!!!!!!` không cải thiện model nhiều
- Giảm noise trong data
- Nhưng vẫn giữ được cảm xúc (không bỏ hết dấu)

**Chuẩn hoá các ký tự đặc biệt:**
```python
"…" → "..."
"–" → "-"
"—" → "-"
""" → '"'
"'" → "'"
```

---

### 3️⃣ Special Token Mapping (Thay Thế Token Đặc Biệt)

**Mục đích:** URL, số, emoji không mang nhiều ý nghĩa sentiment riêng lẻ

#### a) URL Mapping
```python
# Before: "Học tại https://fpt.edu.vn rất tốt"
# After:  "Học tại <URL> rất tốt"
```

**Lợi ích:**
- Model không cần học mỗi URL khác nhau
- Giảm vocabulary
- Tập trung vào context xung quanh

#### b) Number Mapping
```python
# Before: "Điểm của em là 9.5 rất cao"
# After:  "Điểm của em là <NUM> rất cao"

# Before: "Giá 500,000 đồng"
# After:  "Giá <NUM> đồng"
```

**Note:** Con số cụ thể không quan trọng bằng context

#### c) Emoji Mapping
```python
# Before: "Quá vui 😊🎉❤️"
# After:  "Quá vui <EMOJI>"
```

**Lý do:**
- Emoji thể hiện cảm xúc nhưng đa dạng quá
- Map về 1 token để model nhận biết "có cảm xúc được biểu đạt"
- Có thể refine sau bằng emoji-specific model

---

### 4️⃣ Teencode/Slang Handling (Xử Lý Tiếng Lóng)

**Vấn đề:** Người Việt hay dùng teencode/slang trong comments

#### Dictionary Mapping (80+ entries)

```python
# Phủ định
"k"/"ko"/"kg" → "không"
"hong"/"hông" → "không"

# Động từ
"dc"/"đc" → "được"
"lm" → "làm"
"hc"/"hok" → "học"

# Giới từ
"vs"/"vc"/"v" → "với"
"tr"/"trog" → "trong"

# Đại từ
"t"/"tui" → "tôi"
"m"/"mk"/"mik" → "mình"
"ng"/"ngta" → "người"

# Trạng từ
"wa"/"qá" → "quá"
"ms" → "mới"
"dag"/"đag" → "đang"

# Câu hỏi
"ntn" → "như thế nào"
"j"/"gi" → "gì"
```

**Ví dụ thực tế:**
```python
# Before: "k biết ntn nhỉ, t vs bạn m dc học chung ko?"
# After:  "không biết như thế nào nhỉ, tôi với bạn mình được học chung không?"
```

**Impact lên Sentiment:**
- "k" → "không": Rất quan trọng cho phủ định
- "wa" → "quá": Thể hiện mức độ cảm xúc
- Chuẩn hoá giúp model hiểu đúng nghĩa

---

### 5️⃣ Sentence Splitting (Tách Câu)

**Mục đích:** Hỗ trợ error analysis, hiểu context từng câu

```python
# Before: "Thầy dạy hay. Sinh viên rất vui. Nội dung bổ ích."
# After:  ["Thầy dạy hay", "Sinh viên rất vui", "Nội dung bổ ích"]
# Stored: "Thầy dạy hay [SEP] Sinh viên rất vui [SEP] Nội dung bổ ích"
```

**Pattern:** Tách ở dấu `.!?` + khoảng trắng + chữ cái hoa

**Use Cases:**
- Phân tích câu nào gây ra misclassification
- Multi-aspect sentiment (1 review có nhiều khía cạnh)
- Fine-grained analysis

**Note:** Unit phân loại vẫn là **toàn bộ đoạn văn**, không phải từng câu

---

### 6️⃣ Length Filtering (Lọc Độ Dài)

**Config:**
```yaml
min_length: 10   # characters
max_length: 5000 # characters
```

**Loại bỏ:**
```python
# Too short (< 10 chars): "ok", "đc", "k"
# → Không đủ thông tin để phân loại

# Too long (> 5000 chars): Spam, copy-paste dài
# → Có thể là noise
```

**Kết quả:** 32,966 → 32,944 samples (-22 samples, 0.07%)

---

### 7️⃣ Deduplication (Loại Bỏ Trùng Lặp)

**Phương pháp:** Exact matching (nhanh) hoặc Fuzzy matching (chậm hơn)

#### Exact Deduplication (Hiện tại dùng)
```python
# Loại bỏ text giống nhau 100%
df = df.drop_duplicates(subset=['text_processed'])
```

**Ưu điểm:**
- Nhanh (vài giây)
- Loại spam/copy-paste
- Giảm overfitting

#### Fuzzy Deduplication (Optional)
```python
# Loại text giống ~95%
# "Sản phẩm tốt" vs "Sản phẩm rất tốt" → 90% similar
threshold = 0.95
```

**Trade-off:**
- Chậm hơn nhiều (O(n²))
- Có thể loại bỏ variations hợp lệ
- Chỉ dùng khi cần thiết

**Kết quả:** 32,944 → 32,930 samples (-14 duplicates, 0.04%)

---

### 8️⃣ Remove Small Classes (Loại Class Quá Nhỏ)

**Vấn đề:** Stratified split cần ít nhất 6 samples/class

```python
# Example small class:
# (Topic=0.0, Sentiment=3.0): chỉ có 1 sample
# → Không thể split thành train/val/test

min_samples_per_class = 6
```

**Process:**
```python
# Tạo stratification key
df['_stratify'] = df['classification'].astype(str) + '_' + df['sentiment'].astype(str)

# Đếm samples
class_counts = df['_stratify'].value_counts()

# Loại bỏ class < 6 samples
small_classes = class_counts[class_counts < 6]
df = df[~df['_stratify'].isin(small_classes.index)]
```

**Impact:**
- Loại bỏ một số combinations hiếm
- Đảm bảo split được thực hiện thành công
- Trade-off: Mất 1 ít data nhưng cần thiết

---

### 9️⃣ Stratified Split (Chia Tập Dữ Liệu)

**Mục tiêu:** Chia 70/15/15 sao cho **giữ nguyên tỷ lệ** sentiment × topic

#### Tại Sao Stratified?

**Không stratified:**
```
Train: 90% Neutral, 5% Positive, 5% Negative
Test:  30% Neutral, 50% Positive, 20% Negative
→ Model học thiên vị, test không đại diện
```

**Có stratified:**
```
Train: 69% Neutral, 16% Negative, 13% Positive, 2% Mixed
Val:   69% Neutral, 16% Negative, 13% Positive, 2% Mixed
Test:  69% Neutral, 16% Negative, 13% Positive, 2% Mixed
→ Cân bằng, đại diện
```

#### Algorithm

```python
# Step 1: Train vs (Val+Test) - 70/30
train, temp = train_test_split(
    df, 
    test_size=0.30,
    stratify=df['_stratify'],  # (topic, sentiment)
    random_state=42
)

# Step 2: Val vs Test - 50/50 of remaining 30%
val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp['_stratify'],
    random_state=42
)
```

#### Fixed Seed = Reproducibility
```python
seed = 42  # Cố định
```

**Lợi ích:**
- Chạy lại → kết quả giống hệt
- So sánh experiments công bằng
- Debug dễ dàng hơn

#### Kết Quả Split

| Split | Samples | Ratio | Check |
|-------|---------|-------|-------|
| Train | 23,050 | 70.0% | ✅ |
| Val | 4,940 | 15.0% | ✅ |
| Test | 4,940 | 15.0% | ✅ |

---

## 🎯 Đặc Điểm Sentiment Classes

### Class 0.0 - Neutral (Trung Tính)

**Đặc điểm:**
- Chiếm **69.1%** dataset → Dominant class
- Không có cảm xúc rõ ràng positive/negative
- Thường là câu hỏi, mô tả khách quan

**Ví dụ:**
```
✓ "Em muốn hỏi về lịch thi học kỳ này"
✓ "Trường có bao nhiêu khoa?"
✓ "Địa chỉ phòng 101 ở đâu?"
```

**Thách thức:**
- Model dễ bias về class này
- Cần balanced loss hoặc class weights

### Class 1.0 - Positive (Tích Cực)

**Đặc điểm:**
- Chiếm **12.6%** dataset
- Thể hiện sự hài lòng, khen ngợi
- Keywords: "tốt", "hay", "vui", "thích", "cảm ơn"

**Ví dụ:**
```
✓ "Thầy dạy rất hay và dễ hiểu"
✓ "Môi trường học tập tuyệt vời"
✓ "Cảm ơn thầy đã nhiệt tình hướng dẫn"
```

**Indicators:**
- Emoji positive: 😊 🎉 ❤️ 👍
- Từ cường độ: "rất", "quá", "cực kỳ" + positive adj

### Class 2.0 - Negative (Tiêu Cực)

**Đặc điểm:**
- Chiếm **15.8%** dataset
- Thể hiện sự bất mãn, phàn nàn
- Keywords: "tệ", "kém", "thất vọng", "không tốt"

**Ví dụ:**
```
✓ "Thất vọng về cách phục vụ"
✓ "Giảng viên không nhiệt tình"
✓ "Cơ sở vật chất quá kém"
```

**Indicators:**
- Phủ định + positive: "không tốt", "không hài lòng"
- Từ cường độ: "quá", "rất" + negative adj
- Emoji negative: 😞 😠 💔

### Class 3.0 - Mixed (Hỗn Hợp)

**Đặc điểm:**
- Chiếm **2.5%** dataset → ⚠️ **MINOR CLASS**
- Vừa có positive vừa có negative
- Phức tạp nhất để phân loại

**Ví dụ:**
```
✓ "Giảng viên dạy hay nhưng đề thi khó quá"
✓ "Trường đẹp nhưng xa, đi lại bất tiện"
✓ "Học phí rẻ nhưng chất lượng chưa tốt"
```

**Thách thức:**
- Ít data → dễ underfit
- Model cần học được contrast
- Có thể apply augmentation

---

## 🔬 Xử Lý Class Imbalance

### Phát Hiện Imbalance

**Threshold:** Classes < 5% được đánh dấu "minor"

```
Sentiment Distribution:
  ✅ Class 0.0: 69.1% (Neutral) - OK
  ✅ Class 2.0: 15.8% (Negative) - OK
  ✅ Class 1.0: 12.6% (Positive) - OK
  ⚠️ Class 3.0: 2.5% (Mixed) - MINOR ← CHÚ Ý!
```

### Chiến Lược Xử Lý

#### 1. Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight

# Tính weights tự động
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Apply trong model
model.compile(
    loss='sparse_categorical_crossentropy',
    class_weight=class_weights
)
```

**Effect:** Penalty mạnh hơn cho sai lầm trên minor class

#### 2. Oversampling (SMOTE, ADASYN)
```python
from imblearn.over_sampling import SMOTE

# Tạo synthetic samples cho minor class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Lưu ý:** Chỉ apply lên train set, KHÔNG dùng cho val/test

#### 3. Data Augmentation
```python
# Cho class 3.0 (Mixed):
# - Paraphrase
# - Back-translation
# - Synonym replacement
# - Random insertion/deletion
```

#### 4. Ensemble Methods
```python
# Train nhiều models với sampling strategies khác nhau
# Kết hợp predictions để cải thiện minor class
```

---

## 📈 Evaluation Metrics cho Imbalanced Data

### ❌ KHÔNG nên chỉ dùng Accuracy

```python
# Ví dụ: Model luôn predict Neutral (69.1%)
# Accuracy = 69.1% nhưng model vô dụng!
```

### ✅ Nên dùng

#### 1. Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
# Nhìn thấy rõ model classify sai như thế nào
```

#### 2. Per-Class Metrics
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

Output:
```
              precision    recall  f1-score   support

         0.0       0.85      0.92      0.88      3412
         1.0       0.75      0.68      0.71       623
         2.0       0.78      0.73      0.75       781
         3.0       0.60      0.45      0.51       124  ← Chú ý class này!

    accuracy                           0.82      4940
   macro avg       0.75      0.70      0.71      4940
weighted avg       0.81      0.82      0.81      4940
```

**Chú ý:**
- `macro avg`: Trung bình tất cả classes (không quan tâm size)
- `weighted avg`: Trung bình có trọng số (theo size)
- Class 3.0 có F1 thấp nhất → Cần cải thiện

#### 3. F1-Score (Macro/Weighted)
```python
from sklearn.metrics import f1_score

# Macro: Average của tất cả classes
f1_macro = f1_score(y_true, y_pred, average='macro')

# Weighted: Có trọng số theo class size
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

#### 4. ROC-AUC (Multi-class)
```python
from sklearn.metrics import roc_auc_score

# One-vs-Rest
auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
```

---

## 🎯 Tips & Best Practices

### 1. Preprocessing
✅ **DO:**
- Giữ dấu tiếng Việt (crucial!)
- Map URL/NUM/EMOJI nhất quán
- Handle teencode (cải thiện coverage)
- Split data stratified
- Fix random seed

❌ **DON'T:**
- Bỏ dấu tiếng Việt
- Remove tất cả punctuation (mất cảm xúc)
- Over-clean (mất thông tin)
- Split random (không đại diện)

### 2. Handling Imbalance
✅ **DO:**
- Monitor per-class metrics
- Use class weights
- Augment minor classes
- Validate on balanced set

❌ **DON'T:**
- Chỉ nhìn accuracy
- Bỏ qua minor classes
- Oversample val/test sets
- Underestimate impact

### 3. Validation
✅ **DO:**
- Stratified K-fold CV
- Separate validation set
- Check confusion matrix
- Analyze errors per class

❌ **DON'T:**
- Train/test chung
- Leak data giữa splits
- Bỏ qua class distribution

---

## 📊 Output Files

### 1. Processed Data
```
data/processed/
├── train.parquet       # 23,050 samples
├── val.parquet        # 4,940 samples
├── test.parquet       # 4,940 samples
└── statistics.json    # Dataset stats
```

**Columns trong parquet:**
```python
{
    'text': str,                  # Original text
    'text_processed': str,        # Preprocessed text
    'sentiment': float,           # 0.0, 1.0, 2.0, 3.0
    'classification': float,      # Topic 0.0-9.0
    'text_length': int,          # Character count
    'num_words': int,            # Word count
    'num_sentences': int,        # Sentence count
    'sentences': list,           # List of sentences
    'original_split': str        # 'train'/'val'/'test'
}
```

### 2. Split Indices
```
splits/
├── splits.json           # Complete metadata
├── train_indices.json    # List of train indices
├── val_indices.json      # List of val indices
└── test_indices.json     # List of test indices
```

**splits.json format:**
```json
{
    "seed": 42,
    "timestamp": "2025-10-31T...",
    "train_indices": [0, 5, 12, ...],
    "val_indices": [2, 7, 15, ...],
    "test_indices": [1, 3, 8, ...],
    "train_size": 23050,
    "val_size": 4940,
    "test_size": 4940
}
```

### 3. Statistics
```
data/processed/statistics.json
```

Contains:
- Number of samples per split
- Average text length/words/sentences
- Sentiment distribution
- Topic distribution

---

## 🚀 Sử Dụng Trong Training

### Load Processed Data

```python
import pandas as pd

# Load data
train_df = pd.read_parquet('data/processed/train.parquet')
val_df = pd.read_parquet('data/processed/val.parquet')
test_df = pd.read_parquet('data/processed/test.parquet')

# Get features and labels
X_train = train_df['text_processed'].values
y_train_sentiment = train_df['sentiment'].values
y_train_topic = train_df['classification'].values

print(f"Train: {len(X_train)} samples")
print(f"Sentiment classes: {sorted(train_df['sentiment'].unique())}")
print(f"Topic classes: {sorted(train_df['classification'].unique())}")
```

### Tokenization Example

```python
from transformers import AutoTokenizer

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Tokenize
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors='pt'
)
```

### Multi-Task Learning (Sentiment + Topic)

```python
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_sentiments=4, num_topics=10):
        super().__init__()
        self.base = base_model
        
        # Task-specific heads
        self.sentiment_head = nn.Linear(768, num_sentiments)
        self.topic_head = nn.Linear(768, num_topics)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        pooled = outputs.pooler_output
        
        sentiment_logits = self.sentiment_head(pooled)
        topic_logits = self.topic_head(pooled)
        
        return sentiment_logits, topic_logits

# Loss với class weights
sentiment_weights = torch.tensor([1.0, 1.5, 1.3, 3.0])  # Higher for minor class 3.0
criterion_sentiment = nn.CrossEntropyLoss(weight=sentiment_weights)
criterion_topic = nn.CrossEntropyLoss()

# Training loop
for batch in train_loader:
    sentiment_logits, topic_logits = model(batch['input_ids'], batch['attention_mask'])
    
    loss_sentiment = criterion_sentiment(sentiment_logits, batch['sentiment'])
    loss_topic = criterion_topic(topic_logits, batch['topic'])
    
    # Combined loss
    loss = loss_sentiment + 0.5 * loss_topic  # Weight topic task less
    
    loss.backward()
    optimizer.step()
```

---

## 📚 References & Resources

### Vietnamese NLP Tools

1. **PhoBERT** (Recommended)
   ```bash
   pip install transformers
   # vinai/phobert-base or vinai/phobert-large
   ```

2. **VnCoreNLP** (Word segmentation, POS tagging)
   ```bash
   pip install vncorenlp
   ```

3. **Underthesea** (Vietnamese NLP toolkit)
   ```bash
   pip install underthesea
   ```

### Papers

- PhoBERT: "PhoBERT: Pre-trained language models for Vietnamese" (2020)
- Vietnamese Sentiment: "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis" (2018)

### Datasets

- UIT-VSFC: Vietnamese Students' Feedback
- VLSP Sentiment Analysis Dataset
- ViMMRC: Vietnamese Machine Reading Comprehension

---

## 🎓 Summary cho Báo Cáo/Presentation

### Key Points

1. **Dataset:** 32,930 samples tiếng Việt, 4 sentiment classes, 10 topic classes
2. **Imbalance:** Neutral dominant (69%), Mixed minor (2.5%)
3. **Preprocessing:** 9 bước từ raw text → clean data
   - Giữ dấu tiếng Việt (critical!)
   - Map special tokens (URL/NUM/EMOJI)
   - Handle teencode (80+ entries)
   - Stratified split (70/15/15)
4. **Challenges:**
   - Class imbalance → class weights/oversampling
   - Vietnamese-specific → PhoBERT, keep tones
   - Minor class (3.0) → augmentation
5. **Metrics:** Precision/Recall/F1 per class, macro F1, confusion matrix

### Slide Outline

1. **Problem Statement**
   - Vietnamese sentiment classification
   - Multi-class + imbalanced data
   
2. **Dataset Analysis**
   - Distribution visualization
   - Class imbalance identification
   
3. **Preprocessing Pipeline**
   - Flow diagram (9 steps)
   - Vietnamese-specific considerations
   
4. **Handling Imbalance**
   - Class weights
   - Evaluation metrics
   
5. **Results & Discussion**
   - Per-class performance
   - Error analysis

---

## ✅ Checklist

### Preprocessing
- [x] Lowercase + keep Vietnamese tones
- [x] Punctuation normalization
- [x] Special token mapping (URL/NUM/EMOJI)
- [x] Teencode handling
- [x] Sentence splitting
- [x] Length filtering
- [x] Deduplication
- [x] Stratified split (70/15/15)
- [x] Save to parquet

### Quality Checks
- [x] No data leakage between splits
- [x] Balanced distribution across splits
- [x] Fixed seed for reproducibility
- [x] Statistics logged
- [x] Minor classes identified

### Ready for Training
- [x] Clean data available
- [x] Train/val/test splits
- [x] Class weights computed
- [x] Evaluation metrics defined
- [x] Documentation complete

---

**🎉 Pipeline preprocessing hoàn chỉnh và sẵn sàng cho model training!**

*Good luck với bài toán sentiment classification! 🚀*
