# 🎯 Vietnamese Text Analysis - Web Demo

Streamlit web application for Vietnamese sentiment and topic classification.

## 🚀 Quick Start

### 1. Install Dependencies

```powershell

python -m venv .venv

then

venv/Scripts/activate


# From the web/ directory
cd web
pip install -r requirements.txt
```

### 2. Run the App

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌟 Features

### 🎭 Sentiment Analysis
- **4 Classes:** Positive 😊, Negative 😞, Neutral 😐, Mixed 🤔
- **Confidence Scores:** See probability for each sentiment
- **Interactive Charts:** Beautiful Plotly visualizations

### 📑 Topic Classification
- **10 Topics:** General, Product, Service, Quality, Delivery, Price, Support, Experience, Other, Feedback
- **Multi-class Prediction:** Identifies main topic with confidence
- **Distribution View:** See all topic probabilities

### 🔤 Token-Level Analysis
- **Word-by-Word Sentiment:** Each word is color-coded
- **Sentiment Keywords:** Highlights positive, negative, neutral words
- **Interactive Tooltips:** Hover to see sentiment score
- **Token Statistics:** Count of positive/negative/neutral tokens

### 📊 Visualization
- **Confidence Charts:** Bar charts for all class probabilities
- **Color-Coded Tokens:** Visual representation of sentiment per word
- **Preprocessing Details:** Show original vs processed text
- **Export Results:** Download as JSON or CSV

## 📸 Screenshots

### Main Interface
```
┌─────────────────────────────────────────┐
│   🎯 Vietnamese Text Analysis          │
│   Multi-Task Sentiment & Topic         │
├─────────────────────────────────────────┤
│                                         │
│  💬 Enter Your Comment                 │
│  ┌───────────────────────────────────┐ │
│  │ Sản phẩm tốt lắm, ship nhanh!    │ │
│  │                                   │ │
│  └───────────────────────────────────┘ │
│           [🚀 Analyze]                 │
│                                         │
├─────────────────────────────────────────┤
│  🎭 Sentiment: Positive 😊 (85.2%)     │
│  📑 Topic: Product 📦 (78.5%)          │
├─────────────────────────────────────────┤
│  🔤 Token Analysis:                    │
│  [sản phẩm] [tốt] [lắm] [ship] [nhanh]│
│     ⚪      🟢   🟢    ⚪     🟢       │
└─────────────────────────────────────────┘
```

## 🎨 Color Coding

### Sentiment Colors
- 🟢 **Positive:** Green background (`#d4edda`)
- 🔴 **Negative:** Red background (`#f8d7da`)
- ⚪ **Neutral:** Gray background (`#e2e3e5`)
- 🟡 **Mixed:** Yellow background (`#fff3cd`)

### Token Intensity
- Darker = Stronger sentiment signal
- Lighter = Weaker sentiment signal
- Opacity based on confidence score (0.0 - 1.0)

## 📝 Example Comments

### Positive Examples
```
✅ "Sản phẩm tốt lắm, tôi rất hài lòng! 😊"
   → Sentiment: Positive (92%)
   → Topic: Product (88%)

✅ "Shop phục vụ tốt, giao hàng nhanh, cảm ơn!"
   → Sentiment: Positive (95%)
   → Topic: Service (85%)
```

### Negative Examples
```
❌ "Giao hàng chậm quá, thất vọng!"
   → Sentiment: Negative (87%)
   → Topic: Delivery (92%)

❌ "Sản phẩm kém chất lượng, không đáng tiền"
   → Sentiment: Negative (91%)
   → Topic: Quality (89%)
```

### Mixed Examples
```
🤔 "Chất lượng tốt nhưng giá hơi đắt."
   → Sentiment: Mixed (78%)
   → Topic: Quality (45%), Price (38%)

🤔 "Ship nhanh nhưng sản phẩm không như mô tả"
   → Sentiment: Mixed (82%)
   → Topic: Product (55%), Delivery (30%)
```

## 🔧 Customization

### Adding Real Model

Replace `MockModel` in `app.py` with your trained model:

```python
import torch
from transformers import AutoTokenizer, AutoModel

class RealModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = torch.load("path/to/your/model.pt")
        self.model.eval()
    
    def predict_sentiment(self, text: str):
        # Your model inference code
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        # ... process outputs
        return pred_class, probs
```

### Modifying Sentiment Labels

Edit `SENTIMENT_LABELS` dictionary in `app.py`:

```python
SENTIMENT_LABELS = {
    0: "Very Negative 😡",
    1: "Negative 😞",
    2: "Neutral 😐",
    3: "Positive 😊",
    4: "Very Positive 😍"
}
```

### Adding More Topics

Edit `TOPIC_LABELS` dictionary:

```python
TOPIC_LABELS = {
    0: "Electronics 📱",
    1: "Fashion 👗",
    2: "Food 🍔",
    # ... add more
}
```

### Changing Keywords

Modify keyword sets for better token analysis:

```python
POSITIVE_KEYWORDS = {
    'excellent', 'amazing', 'wonderful',
    # ... add Vietnamese words
}
```

## 📊 Architecture

```
app.py
├── MockModel (or RealModel)
│   ├── preprocess_text()
│   ├── predict_sentiment()
│   ├── predict_topic()
│   └── analyze_tokens()
├── UI Components
│   ├── Text Input Area
│   ├── Prediction Display
│   ├── Token Visualization
│   └── Charts (Plotly)
└── Export Functions
    ├── JSON export
    └── CSV export
```

## 🎯 Use Cases

### 1. Social Media Monitoring
- Analyze customer comments on Facebook/Instagram
- Track sentiment trends over time
- Identify product/service issues

### 2. E-commerce Reviews
- Classify Shopee/Lazada reviews
- Detect fake reviews (mixed sentiment patterns)
- Prioritize negative feedback

### 3. Customer Support
- Auto-route complaints to relevant departments
- Sentiment-based ticket prioritization
- Topic-based assignment

### 4. Market Research
- Analyze competitor reviews
- Identify customer pain points
- Track sentiment by topic

## 🚀 Deployment

### Local Development
```powershell
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Deploy from GitHub repo
4. Set `web/app.py` as main file

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY ../src ./src
COPY ../configs ./configs

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## 📱 Mobile Responsive

The app is fully responsive and works on:
- 📱 Mobile phones
- 📱 Tablets
- 💻 Desktop browsers

## 🔒 Security Notes

For production deployment:
1. **Rate Limiting:** Add request throttling
2. **Input Validation:** Sanitize user input
3. **Model Protection:** Don't expose model weights
4. **HTTPS:** Use SSL certificates
5. **Authentication:** Add user login if needed

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Use different port
streamlit run app.py --server.port=8502
```

### Module Not Found
```powershell
# Make sure you're in the right directory
cd web
pip install -r requirements.txt
```

### Preprocessing Error
```powershell
# Check configs/config.yaml exists
# Check src/preprocess.py exists
```

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Docs](https://plotly.com/python/)
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT)

## 🎓 Project Info

- **Course:** Text Mining & Generation (TMG)
- **Semester:** Fall 2025
- **University:** FPT University
- **Model:** Multi-Task BERT for Vietnamese Text

---

**Made with ❤️ for Vietnamese NLP**
