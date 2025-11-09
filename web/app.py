"""
🎯 Vietnamese Text Analysis Demo
Multi-Task Sentiment & Topic Classification

Streamlit Web App for TMG Fall 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import re
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Try to import PyTorch and Transformers (for real model)
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch/Transformers not installed. Using keyword-based model.")

# Add parent directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Import preprocessing module
try:
    from src.preprocess import VietnameseTextPreprocessor, load_config
    PREPROCESS_AVAILABLE = True
except ImportError:
    try:
        from preprocess import VietnameseTextPreprocessor, load_config
        PREPROCESS_AVAILABLE = True
    except ImportError:
        st.warning("⚠️ Cannot import preprocessing module. Using basic text processing.")
        PREPROCESS_AVAILABLE = False
        VietnameseTextPreprocessor = None
        load_config = None

# Page configuration
st.set_page_config(
    page_title="Vietnamese Text Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .sentiment-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .sentiment-mixed {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .token-box {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem;
        border-radius: 0.5rem;
        font-size: 1rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .token-box:hover {
        transform: scale(1.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# Sentiment & Topic mappings (from actual dataset)
SENTIMENT_LABELS = {
    0.0: "Neutral �",
    1.0: "Positive �",
    2.0: "Negative �",
    3.0: "Mixed 🤔"
}

SENTIMENT_COLORS = {
    0.0: "#e2e3e5",  # Neutral - gray
    1.0: "#d4edda",  # Positive - green
    2.0: "#f8d7da",  # Negative - red
    3.0: "#fff3cd"   # Mixed - yellow
}

# Topic labels from dataset (0.0 - 9.0)
TOPIC_LABELS = {
    0.0: "Spam 📧",
    1.0: "News 📰",
    2.0: "Academic 📚",
    3.0: "Other 💬",
    4.0: "Service 🛎️",
    5.0: "Jobs & Recruitment 💼",
    6.0: "Personal Affairs �",
    7.0: "Social Affairs �",
    8.0: "Help & Share 🤝",
    9.0: "Club & Events 🎉"
}


# Vietnamese sentiment keywords for token analysis
POSITIVE_KEYWORDS = {
    'tốt', 'hay', 'đẹp', 'tuyệt', 'xuất sắc', 'hoàn hảo', 'thích', 'yêu', 'ưng', 
    'ok', 'oke', 'ngon', 'chất', 'mantap', 'xịn', 'mượt', 'chuẩn', 'đáng', 
    'hài lòng', 'hài', 'lòng', 'vui', 'thích thú', 'tươi', 'sạch', 'nhanh', 'tiện', 'dễ',
    'cảm ơn', 'thank', 'thanks', 'cám ơn', 'ủng hộ', 'recommend', 'tuyệt vời',
    'xuất sắc', 'ưng ý', 'hợp lý', 'đáng tiền', 'quality', 'perfect', 'excellent',
    'tuyệt vời', 'rất', 'lắm', 'quá', 'cực kỳ', 'siêu', 'top', 'best'
}

NEGATIVE_KEYWORDS = {
    'tệ', 'kém', 'dở', 'thất vọng', 'tồi', 'chán', 'ghét', 'không tốt', 
    'xấu', 'lỗi', 'hỏng', 'sai', 'lừa', 'giả', 'fake', 'scam', 'gian lận',
    'chậm', 'lâu', 'thiếu', 'mất', 'hư', 'vỡ', 'hết', 'bẩn',
    'thối', 'ôi', 'khó', 'phức tạp', 'thô', 'ức chế', 'không thích',
    'tồi tệ', 'kinh khủng', 'bad', 'terrible', 'worst', 'horrible'
}

NEUTRAL_KEYWORDS = {
    'bình thường', 'thường', 'được', 'cũng được', 'tạm', 'ổn', 'không sao',
    'như vậy', 'vậy', 'thế', 'nhỉ', 'à', 'ừ', 'uh', 'hả', 'sao', 'gì', 'tôi', 'em'
}


# BERT Multi-Task Architecture (from tmg-bert.ipynb)
if TORCH_AVAILABLE:
    class BERTMultiTaskArchitecture(nn.Module):
        """
        BERT Multi-Task Architecture from notebook
        Backbone chung + 2 MLP heads riêng biệt
        """
        def __init__(self, backbone_model, num_sent_classes, num_topic_classes):
            super(BERTMultiTaskArchitecture, self).__init__()
            
            # 1. Backbone chung
            self.bert = AutoModel.from_pretrained(backbone_model)
            hidden_size = self.bert.config.hidden_size  # 768 for PhoBERT
            
            # 2. Dropout
            self.dropout = nn.Dropout(0.1)
            
            # 3. Head-S (Sentiment)
            self.head_s = nn.Linear(hidden_size, num_sent_classes)
            
            # 4. Head-T (Topic)
            self.head_t = nn.Linear(hidden_size, num_topic_classes)
        
        def forward(self, input_ids, attention_mask):
            # Get output from backbone
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get [CLS] token embedding
            cls_output = outputs.last_hidden_state[:, 0, :]
            
            # Apply dropout
            pooled_output = self.dropout(cls_output)
            
            # Pass through both heads
            logits_s = self.head_s(pooled_output)
            logits_t = self.head_t(pooled_output)
            
            return logits_s, logits_t


class BERTMultiTaskModel:
    """
    BERT Multi-Task Model for Vietnamese Text Analysis
    Architecture from tmg-bert.ipynb
    """
    
    def __init__(self):
        self.preprocessor = None
        self.dataset = None
        self.model = None
        self.tokenizer = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        # Model configuration from notebook
        self.MODEL_NAME = 'vinai/phobert-base'
        self.MAX_LENGTH = 256
        self.N_SENT_CLASSES = 4  # From dataset: 0.0, 1.0, 2.0, 3.0
        self.N_TOPIC_CLASSES = 10  # From dataset: 0.0-9.0
        
        # Load preprocessor (optional)
        if PREPROCESS_AVAILABLE and VietnameseTextPreprocessor is not None:
            try:
                config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
                config = load_config(str(config_path))
                self.preprocessor = VietnameseTextPreprocessor(config['preprocessing'])
                st.success("✅ Preprocessor loaded")
            except Exception as e:
                st.warning(f"⚠️ Using simplified preprocessing: {e}")
        
        # Load real dataset for examples
        try:
            self.load_dataset()
            st.success(f"✅ Dataset loaded: {len(self.dataset)} samples")
        except Exception as e:
            st.warning(f"⚠️ Could not load dataset: {e}")
        
        # Try to load trained model
        try:
            self.load_model()
            st.success("✅ BERT Multi-Task model loaded")
        except Exception as e:
            st.warning(f"⚠️ Using keyword-based model: {e}")
    
    def load_dataset(self):
        """Load TMG dataset from CSV files"""
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent / 'TMG_Dataset',  # From web/ to root
            Path(__file__).parent / 'TMG_Dataset',  # Same directory as app.py
            Path('TMG_Dataset'),  # Relative to current working directory
            Path('../TMG_Dataset')  # One level up
        ]
        
        dataset_path = None
        for path in possible_paths:
            if path.exists() and (path / 'train_set.csv').exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            raise FileNotFoundError("Cannot find TMG_Dataset directory with train_set.csv")
        
        # Load all splits
        train_df = pd.read_csv(dataset_path / 'train_set.csv')
        val_df = pd.read_csv(dataset_path / 'val_set.csv')
        test_df = pd.read_csv(dataset_path / 'test_set.csv')
        
        # Combine
        self.dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    def load_model(self):
        """Load trained BERT Multi-Task model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Check for saved model
        model_path = Path(__file__).parent.parent / 'models' / 'bert_multitask_best.pt'
        
        if model_path.exists():
            # Load trained model
            st.info(f"Loading model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = BERTMultiTaskArchitecture(
                self.MODEL_NAME,
                self.N_SENT_CLASSES,
                self.N_TOPIC_CLASSES
            ).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            # Initialize tokenizer for keyword-based prediction
            st.warning(f"Model file not found at {model_path}. Using keyword-based predictions.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            except Exception:
                pass
    
    def get_random_examples(self, n=10):
        """Get random examples from dataset"""
        if self.dataset is not None:
            return self.dataset.sample(n=min(n, len(self.dataset)))
        return None
    
    def get_examples_by_sentiment(self, sentiment: float, n=5):
        """Get examples for specific sentiment"""
        if self.dataset is not None:
            examples = self.dataset[self.dataset['sentiment'] == sentiment]
            return examples.sample(n=min(n, len(examples)))
        return None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        if self.preprocessor:
            # preprocessor.preprocess_text returns (processed_text, sentences)
            processed_text, _ = self.preprocessor.preprocess_text(text)
            return processed_text
        else:
            # Simple preprocessing
            text = text.lower().strip()
            return text
    
    def predict_sentiment(self, text: str) -> Tuple[float, Dict[float, float]]:
        """
        Predict sentiment with confidence scores
        Uses BERT model if available, otherwise keyword-based
        """
        # Try BERT model first
        if self.model is not None and self.tokenizer is not None:
            try:
                return self._predict_with_bert(text)
            except Exception as e:
                st.warning(f"BERT prediction failed: {e}. Falling back to keywords.")
        
        # Fallback to keyword-based prediction
        return self._predict_with_keywords(text)
    
    def _predict_with_bert(self, text: str) -> Tuple[float, Dict[float, float]]:
        """Predict using trained BERT model"""
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=self.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            logits_s, logits_t = self.model(input_ids, attention_mask)
            
            # Get probabilities
            probs_s = torch.softmax(logits_s, dim=1).cpu().numpy()[0]
            probs_t = torch.softmax(logits_t, dim=1).cpu().numpy()[0]
        
        # Get predictions
        pred_s = int(torch.argmax(logits_s, dim=1).cpu().numpy()[0])
        pred_t = int(torch.argmax(logits_t, dim=1).cpu().numpy()[0])
        
        # Convert to float keys (matching dataset format)
        sentiment_probs = {float(i): float(probs_s[i]) for i in range(self.N_SENT_CLASSES)}
        topic_probs = {float(i): float(probs_t[i]) for i in range(self.N_TOPIC_CLASSES)}
        
        # Store topic prediction for later use
        self._last_topic_pred = float(pred_t)
        self._last_topic_probs = topic_probs
        
        return float(pred_s), sentiment_probs
    
    def _predict_with_keywords(self, text: str) -> Tuple[float, Dict[float, float]]:
        """
        Predict sentiment using keyword matching (fallback)
        Returns: (predicted_class, probability_dict)
        """
        text_lower = text.lower()
        
        # Check for negation words
        negation_words = {'không', 'chẳng', 'chả', 'ko', 'k', 'hong', 'hông', 'chưa'}
        has_negation = any(neg in text_lower.split() for neg in negation_words)
        
        tokens = re.findall(r'\w+', text_lower)
        
        pos_count = sum(1 for word in tokens if word in POSITIVE_KEYWORDS)
        neg_count = sum(1 for word in tokens if word in NEGATIVE_KEYWORDS)
        neu_count = sum(1 for word in tokens if word in NEUTRAL_KEYWORDS)
        
        # Handle negation: flip positive to negative
        if has_negation and pos_count > 0:
            # "không tốt" -> negative
            neg_count += pos_count
            pos_count = max(0, pos_count - 2)  # Reduce positive count
        
        # Determine sentiment
        if pos_count > 0 and neg_count > 0:
            pred_class = 3.0  # Mixed
            confidence = min(0.75, 0.45 + (pos_count + neg_count) * 0.05)
            probs = {0.0: 0.05, 1.0: pos_count * 0.15, 2.0: neg_count * 0.15, 3.0: confidence}
        elif pos_count > neg_count and pos_count > 0:
            pred_class = 1.0  # Positive
            confidence = min(0.90, 0.55 + pos_count * 0.10)
            probs = {0.0: 0.05, 1.0: confidence, 2.0: 0.03, 3.0: max(0.02, 1.0 - confidence - 0.08)}
        elif neg_count > pos_count and neg_count > 0:
            pred_class = 2.0  # Negative
            confidence = min(0.90, 0.55 + neg_count * 0.10)
            probs = {0.0: 0.05, 1.0: 0.03, 2.0: confidence, 3.0: max(0.02, 1.0 - confidence - 0.08)}
        else:
            pred_class = 0.0  # Neutral
            probs = {0.0: 0.70, 1.0: 0.10, 2.0: 0.10, 3.0: 0.10}
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        return pred_class, probs
    
    def predict_topic(self, text: str) -> Tuple[float, Dict[float, float]]:
        """
        Predict topic with confidence scores
        If BERT was used, return cached results
        """
        # If we have cached BERT prediction, return it
        if hasattr(self, '_last_topic_pred'):
            pred = self._last_topic_pred
            probs = self._last_topic_probs
            # Clear cache
            delattr(self, '_last_topic_pred')
            delattr(self, '_last_topic_probs')
            return pred, probs
        
        # Otherwise use keyword-based topic classification
        text_lower = text.lower()
        
        # Topic keywords
        topic_keywords = {
            0.0: {'spam', 'quảng cáo', 'sale', 'khuyến mãi', 'giảm giá', 'mua ngay'},
            1.0: {'tin', 'news', 'báo', 'thông tin', 'sự kiện', 'mới'},
            2.0: {'học', 'thi', 'môn', 'điểm', 'giảng viên', 'giáo viên', 'sinh viên', 'trường', 'lớp', 'bài', 'đại học'},
            3.0: {'khác', 'other', 'gì', 'sao', 'thế', 'nhỉ', 'à'},
            4.0: {'dịch vụ', 'service', 'hỗ trợ', 'cung cấp', 'chăm sóc', 'phục vụ'},
            5.0: {'tuyển', 'recruitment', 'việc làm', 'job', 'ứng tuyển', 'cv', 'phỏng vấn', 'intern'},
            6.0: {'cá nhân', 'tôi', 'mình', 'em', 'personal', 'riêng tư', 'gia đình'},
            7.0: {'xã hội', 'social', 'cộng đồng', 'society', 'văn hóa', 'chính trị'},
            8.0: {'giúp', 'help', 'share', 'chia sẻ', 'hỏi', 'cần', 'ai', 'giúp đỡ'},
            9.0: {'club', 'event', 'sự kiện', 'câu lạc bộ', 'hoạt động', 'tham gia', 'đăng ký'}
        }
        
        # Count matches for each topic
        topic_scores = {}
        for topic_id, keywords in topic_keywords.items():
            score = sum(1 for word in keywords if word in text_lower)
            topic_scores[topic_id] = score
        
        # Find best matching topic
        max_score = max(topic_scores.values())
        
        if max_score > 0:
            # Found keyword matches
            pred_class = max(topic_scores, key=topic_scores.get)
            base_confidence = min(0.60, 0.30 + max_score * 0.15)
        else:
            # No clear match - default to "Other"
            pred_class = 3.0
            base_confidence = 0.35
        
        # Build probability distribution
        probs = {}
        for topic_id in range(10):
            if topic_id == pred_class:
                probs[float(topic_id)] = base_confidence
            elif topic_scores.get(float(topic_id), 0) > 0:
                # Has some keywords
                probs[float(topic_id)] = min(0.15, topic_scores[float(topic_id)] * 0.10)
            else:
                probs[float(topic_id)] = 0.05
        
        # Normalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        return pred_class, probs
    
    def analyze_tokens(self, text: str) -> List[Dict[str, any]]:
        """
        Analyze each token and assign sentiment
        Returns: List of {token, sentiment, score}
        """
        # Simple word tokenization
        tokens = re.findall(r'\w+', text.lower())
        
        token_analysis = []
        for token in tokens:
            # Determine token sentiment
            if token in POSITIVE_KEYWORDS:
                sentiment = 'positive'
                score = 0.8
            elif token in NEGATIVE_KEYWORDS:
                sentiment = 'negative'
                score = 0.8
            elif token in NEUTRAL_KEYWORDS:
                sentiment = 'neutral'
                score = 0.6
            else:
                sentiment = 'neutral'
                score = 0.3
            
            token_analysis.append({
                'token': token,
                'sentiment': sentiment,
                'score': score
            })
        
        return token_analysis


@st.cache_resource
def load_model():
    """Load model (cached)"""
    with st.spinner("🔄 Loading model..."):
        model = BERTMultiTaskModel()
    return model


def display_token_analysis(tokens: List[Dict[str, any]]):
    """Display tokens with sentiment coloring"""
    
    st.markdown("### 🔤 Token-level Sentiment Analysis")
    st.markdown("*Hover over each word to see its sentiment contribution*")
    
    # Create HTML for tokens
    html_tokens = []
    
    for token_info in tokens:
        token = token_info['token']
        sentiment = token_info['sentiment']
        score = token_info['score']
        
        # Color based on sentiment
        if sentiment == 'positive':
            bg_color = f"rgba(76, 175, 80, {score})"  # Green
            text_color = "#155724"
        elif sentiment == 'negative':
            bg_color = f"rgba(244, 67, 54, {score})"  # Red
            text_color = "#721c24"
        else:
            bg_color = f"rgba(158, 158, 158, {score})"  # Gray
            text_color = "#383d41"
        
        html_tokens.append(
            f'<span class="token-box" style="background-color: {bg_color}; color: {text_color};" '
            f'title="{sentiment.capitalize()} (score: {score:.2f})">{token}</span>'
        )
    
    # Display tokens
    st.markdown(
        '<div style="line-height: 3rem; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">' 
        + ' '.join(html_tokens) + 
        '</div>',
        unsafe_allow_html=True
    )
    
    # Token statistics
    st.markdown("#### 📊 Token Statistics")
    col1, col2, col3 = st.columns(3)
    
    positive_tokens = [t for t in tokens if t['sentiment'] == 'positive']
    negative_tokens = [t for t in tokens if t['sentiment'] == 'negative']
    neutral_tokens = [t for t in tokens if t['sentiment'] == 'neutral']
    
    with col1:
        st.metric("✅ Positive Tokens", len(positive_tokens))
        if positive_tokens:
            st.caption(f"Words: {', '.join([t['token'] for t in positive_tokens[:5]])}")
    
    with col2:
        st.metric("❌ Negative Tokens", len(negative_tokens))
        if negative_tokens:
            st.caption(f"Words: {', '.join([t['token'] for t in negative_tokens[:5]])}")
    
    with col3:
        st.metric("⚪ Neutral Tokens", len(neutral_tokens))


def create_probability_chart(probs: Dict[float, float], labels: Dict[float, str], title: str):
    """Create interactive probability bar chart"""
    
    # Sort by probability
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    labels_list = [labels[k] for k, _ in sorted_items]
    probs_list = [v * 100 for _, v in sorted_items]
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=probs_list,
            y=labels_list,
            orientation='h',
            marker=dict(
                color=probs_list,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{p:.1f}%' for p in probs_list],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Vietnamese Text Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Task Sentiment & Topic Classification | TMG Fall 2025</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Model:** BERT Multi-Task (PhoBERT)
        **Dataset:** TMG Fall 2025
        
        **Architecture:**
        - Backbone: vinai/phobert-base
        - Max Length: 256 tokens
        - Multi-Task Learning
        
        **Tasks:**
        - 🎭 Sentiment (4 classes)
        - 📑 Topic (10 classes)
        
        **Dataset:**
        - Total: 32,930 samples
        - Train: 70% | Val: 15% | Test: 15%
        """)
        
        st.markdown("### 📝 Example Comments")
        st.markdown("**Click to try:**")
        
        # Get real examples from dataset
        if model.dataset is not None:
            # Sample diverse examples
            sample_examples = model.dataset.sample(n=8, random_state=42)
            
            for idx, row in sample_examples.iterrows():
                text_preview = row['text'][:45] + "..." if len(row['text']) > 45 else row['text']
                sent_label = SENTIMENT_LABELS[row['sentiment']]
                if st.button(f"📌 {text_preview}", key=f"ex_{idx}"):
                    st.session_state.example_text = row['text']
                    st.caption(f"Actual: {sent_label}")
        else:
            # Fallback examples
            examples = [
                "Sản phẩm tốt lắm, tôi rất hài lòng! 😊",
                "Giao hàng chậm quá, thất vọng!",
                "Bình thường thôi, không có gì đặc biệt.",
                "Chất lượng tốt nhưng giá hơi đắt."
            ]
            
            for i, ex in enumerate(examples):
                if st.button(f"📌 {ex[:35]}...", key=f"fallback_ex_{i}"):
                    st.session_state.example_text = ex
        
        # Model status
        st.markdown("---")
        st.markdown("### ⚡ Model Status")
        if model.model is not None:
            st.success("✅ BERT Model Loaded")
            st.caption(f"Device: {model.device}")
        else:
            st.warning("⚠️ Using Keyword-based Model")
            st.caption("Train BERT model for better accuracy")
    
    # Main content
    st.markdown("## 💬 Enter Your Comment")
    
    # Get example text if selected
    default_text = st.session_state.get('example_text', '')
    
    # Text input
    user_input = st.text_area(
        "Type or paste a Vietnamese social media comment:",
        value=default_text,
        height=150,
        placeholder="Ví dụ: Sản phẩm này tốt lắm, giá cả hợp lý, ship nhanh. Tôi rất hài lòng!",
        help="Enter Vietnamese text from social media (Facebook, Shopee, TikTok, etc.)"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Analysis results
    if analyze_button and user_input.strip():
        with st.spinner("🔍 Analyzing..."):
            # Preprocess
            processed_text = model.preprocess_text(user_input)
            
            # Predictions
            sentiment_pred, sentiment_probs = model.predict_sentiment(processed_text)
            topic_pred, topic_probs = model.predict_topic(processed_text)
            
            # Token analysis
            token_analysis = model.analyze_tokens(user_input)
        
        st.success("✅ Analysis Complete!")
        
        # Display results
        st.markdown("---")
        
        # Main predictions
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 1rem; color: white; margin-bottom: 1rem;'>
                <h2 style='margin: 0; color: white;'>🎭 Sentiment</h2>
                <h1 style='margin: 0.5rem 0; font-size: 2.5rem; color: white;'>{SENTIMENT_LABELS[sentiment_pred]}</h1>
                <p style='margin: 0; font-size: 1.2rem; opacity: 0.9;'>Confidence: {sentiment_probs[sentiment_pred]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sentiment probability chart
            st.plotly_chart(
                create_probability_chart(sentiment_probs, SENTIMENT_LABELS, "Sentiment Confidence"),
                use_container_width=True
            )
        
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        border-radius: 1rem; color: white; margin-bottom: 1rem;'>
                <h2 style='margin: 0; color: white;'>📑 Topic</h2>
                <h1 style='margin: 0.5rem 0; font-size: 2.5rem; color: white;'>{TOPIC_LABELS[topic_pred]}</h1>
                <p style='margin: 0; font-size: 1.2rem; opacity: 0.9;'>Confidence: {topic_probs[topic_pred]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Topic probability chart
            st.plotly_chart(
                create_probability_chart(topic_probs, TOPIC_LABELS, "Topic Confidence"),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Token analysis
        display_token_analysis(token_analysis)
        
        st.markdown("---")
        
        # Text preprocessing details
        with st.expander("🔧 Preprocessing Details"):
            st.markdown("### Original Text")
            st.code(user_input, language=None)
            
            st.markdown("### Processed Text")
            st.code(processed_text, language=None)
            
            st.markdown("### Text Statistics")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Characters", len(user_input))
            with stats_col2:
                st.metric("Words", len(user_input.split()))
            with stats_col3:
                st.metric("Tokens", len(token_analysis))
            with stats_col4:
                st.metric("Sentences", user_input.count('.') + user_input.count('!') + user_input.count('?') + 1)
        
        # Download results
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        results_dict = {
            'input_text': user_input,
            'processed_text': processed_text,
            'sentiment': {
                'prediction': SENTIMENT_LABELS[sentiment_pred],
                'confidence': float(sentiment_probs[sentiment_pred]),
                'all_probabilities': {SENTIMENT_LABELS[k]: float(v) for k, v in sentiment_probs.items()}
            },
            'topic': {
                'prediction': TOPIC_LABELS[topic_pred],
                'confidence': float(topic_probs[topic_pred]),
                'all_probabilities': {TOPIC_LABELS[k]: float(v) for k, v in topic_probs.items()}
            },
            'token_analysis': token_analysis
        }
        
        results_json = json.dumps(results_dict, ensure_ascii=False, indent=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download JSON",
                data=results_json,
                file_name="analysis_results.json",
                mime="application/json"
            )
        
        with col2:
            # CSV format
            csv_data = f"Text,Sentiment,Sentiment_Confidence,Topic,Topic_Confidence\n"
            csv_data += f'"{user_input}",{SENTIMENT_LABELS[sentiment_pred]},{sentiment_probs[sentiment_pred]:.4f},{TOPIC_LABELS[topic_pred]},{topic_probs[topic_pred]:.4f}'
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="analysis_results.csv",
                mime="text/csv"
            )
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎓 <strong>TMG Fall 2025</strong> - Text Mining & Generation</p>
        <p>🏫 FPT University | Multi-Task BERT for Vietnamese Text Analysis</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
