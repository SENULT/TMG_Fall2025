# TMG Fall 2025 - Text Mining & Generation

Vietnamese Text Classification Pipeline with Advanced Preprocessing, EDA, and Version Control

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Versioning](#data-versioning)
- [Experiment Tracking](#experiment-tracking)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete data preprocessing pipeline for Vietnamese text classification, including:

- ✅ Vietnamese-specific text preprocessing (keep tones, teencode handling)
- ✅ Special token mapping (URL, NUM, EMOJI)
- ✅ Deduplication (~95% threshold)
- ✅ Sentence splitting for error analysis
- ✅ Stratified split (70/15/15) by topic × sentiment
- ✅ Comprehensive EDA with visualizations
- ✅ DVC/Git-LFS for data versioning
- ✅ W&B/MLflow for experiment tracking
- ✅ Robustness and fairness checks

## ✨ Features

### Text Preprocessing
- **Lowercase normalization** while preserving Vietnamese tones
- **Punctuation normalization** (limit consecutive punctuation)
- **Special tokens:** Map URLs, numbers, and emojis
- **Teencode handling:** Convert Vietnamese slang to standard forms
- **Sentence splitting:** For error analysis and fine-grained processing
- **Deduplication:** Remove ~95% similar texts using fuzzy matching
- **Length filtering:** Remove too short/long texts

### Data Processing
- **Stratified splitting:** 70/15/15 ratio by (topic, sentiment)
- **Class balance check:** Identify minority classes (< 5%)
- **Multi-aspect detection:** Analyze sentiment-topic combinations
- **Statistics tracking:** Length, word count, sentence count
- **Fixed seed:** Reproducible splits (seed=42)

### Robustness & Fairness
- Replace special tokens and test model stability
- Punctuation permutation tests
- Small class bias detection
- Cross-validation of label distributions

### Version Control & Tracking
- **DVC:** Track data, models, and pipelines
- **Git-LFS:** For large files (alternative to DVC)
- **W&B:** Experiment tracking and visualization
- **MLflow:** Alternative tracking system

## 📁 Project Structure

```
TMG_Fall2025/
├── configs/
│   ├── config.yaml              # Main configuration
│   └── teencode_dict.json       # Vietnamese teencode dictionary
├── data/
│   ├── raw/                     # Raw CSV files (DVC tracked)
│   └── processed/               # Processed Parquet files (DVC tracked)
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       └── statistics.json
├── splits/
│   ├── splits.json              # Split indices with metadata
│   ├── train_indices.json
│   ├── val_indices.json
│   └── test_indices.json
├── src/
│   ├── preprocess.py            # Core preprocessing module
│   ├── data_pipeline.py         # Data processing pipeline
│   └── tracking.py              # W&B/MLflow integration
├── notebooks/
│   ├── EDA.py                   # Exploratory data analysis
│   └── figures/                 # EDA visualizations
│       ├── sentiment_distribution.png
│       ├── topic_distribution.png
│       ├── text_length_analysis.png
│       ├── sentiment_by_topic.png
│       └── eda_summary.json
├── logs/
│   └── preprocessing.log        # Processing logs
├── models/                      # Trained models (future)
├── run_pipeline.py              # Main pipeline runner
├── requirements.txt             # Python dependencies
├── DATA_CARD.md                 # Dataset documentation
├── DVC_SETUP.md                 # DVC setup guide
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd TMG_Fall2025
```

### 2. Create Virtual Environment

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install DVC (Optional but Recommended)

```bash
pip install dvc
dvc init
```

### 5. Setup W&B (Optional)

```bash
wandb login
```

## ⚡ Quick Start

### Run Complete Pipeline

```bash
python run_pipeline.py --run-eda
```

This will:
1. Load raw data from `data/raw/`
2. Preprocess texts (Vietnamese-specific)
3. Perform stratified split (70/15/15)
4. Save processed data to `data/processed/`
5. Run EDA and generate figures
6. Log to W&B/MLflow (if enabled)

### Run Pipeline Without Tracking

```bash
python run_pipeline.py --skip-tracking --run-eda
```

### Run Only Preprocessing

```bash
python run_pipeline.py --skip-tracking
```

### Run Only EDA

```bash
cd notebooks
python EDA.py
```

## 📖 Usage

### Custom Configuration

Edit `configs/config.yaml` to customize:

```yaml
preprocessing:
  lowercase: true
  map_urls: true
  map_numbers: true
  dedup_threshold: 0.95
  # ... more settings

splitting:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify_by: ["classification", "sentiment"]
```

### Load Processed Data

```python
import pandas as pd

# Load parquet files
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')

# View processed text
print(train[['text_processed', 'sentiment', 'classification']].head())
```

### Use Preprocessing Module

```python
from src.preprocess import VietnameseTextPreprocessor, load_config

# Load config
config = load_config('configs/config.yaml')

# Initialize preprocessor
preprocessor = VietnameseTextPreprocessor(config)

# Preprocess text
text = "Xin chào! Tôi đang học tại https://fpt.edu.vn 😊"
processed, sentences = preprocessor.preprocess_text(text, keep_sentences=True)

print("Original:", text)
print("Processed:", processed)
print("Sentences:", sentences)
```

### Load Split Indices

```python
import json

with open('splits/splits.json', 'r') as f:
    splits = json.load(f)

print(f"Train size: {splits['train_size']}")
print(f"Val size: {splits['val_size']}")
print(f"Test size: {splits['test_size']}")
print(f"Seed: {splits['seed']}")
```

## ⚙️ Configuration

Main configuration file: `configs/config.yaml`

Key sections:
- **project:** Name, version, seed
- **paths:** Data directories
- **preprocessing:** Text processing settings
- **splitting:** Train/val/test ratios
- **class_imbalance:** Threshold for minority classes
- **robustness:** Augmentation tests
- **tracking:** W&B/MLflow settings
- **eda:** Visualization settings

See comments in `config.yaml` for details.

## 💾 Data Versioning

### Setup DVC

See detailed instructions in `DVC_SETUP.md`

```bash
# Initialize DVC
dvc init

# Track data
dvc add data/raw
dvc add data/processed

# Commit DVC files
git add data/raw.dvc data/processed.dvc .gitignore
git commit -m "Track data with DVC"

# Setup remote (optional)
dvc remote add -d myremote /path/to/storage

# Push to remote
dvc push
```

### Pull Data

```bash
dvc pull
```

## 📊 Experiment Tracking

### Weights & Biases (W&B)

Enable in `configs/config.yaml`:

```yaml
tracking:
  use_wandb: true
  wandb:
    project: "tmg-fall2025"
    entity: "your-username"
    tags: ["preprocessing", "vietnamese"]
```

Run with tracking:

```bash
python run_pipeline.py --run-eda
```

View results at: https://wandb.ai/your-username/tmg-fall2025

### MLflow (Alternative)

Enable in `configs/config.yaml`:

```yaml
tracking:
  use_mlflow: true
  mlflow:
    tracking_uri: "mlruns"
    experiment_name: "TMG_Preprocessing"
```

View results:

```bash
mlflow ui
```

## 📚 Documentation

- **[DATA_CARD.md](DATA_CARD.md):** Complete dataset documentation
- **[DVC_SETUP.md](DVC_SETUP.md):** Data versioning setup guide
- **[notebooks/figures/](notebooks/figures/):** EDA visualizations

### Key Documents

1. **Data Card:** Dataset description, preprocessing steps, usage examples
2. **DVC Setup:** Version control for data and models
3. **EDA Results:** Statistical analysis and visualizations

## 🔍 EDA Results

After running EDA, check:

- `notebooks/figures/sentiment_distribution.png` - Sentiment class distribution
- `notebooks/figures/topic_distribution.png` - Topic class distribution
- `notebooks/figures/text_length_analysis.png` - Text length statistics
- `notebooks/figures/sentiment_by_topic.png` - Cross-tabulation heatmap
- `notebooks/figures/eda_summary.json` - Summary statistics

## 🧪 Robustness Checks

Implemented checks:
- ✅ Special token replacement tests
- ✅ Punctuation permutation tests
- ✅ Class imbalance detection
- ✅ Small class bias monitoring

## 📝 TODO

- [ ] Add word segmentation (using `pyvi` or `underthesea`)
- [ ] Implement more sophisticated deduplication (e.g., LSH, embeddings)
- [ ] Add data augmentation techniques
- [ ] Integrate with model training pipeline
- [ ] Add more teencode entries
- [ ] Implement cross-validation splits

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is for educational purposes (FPT University - Text Mining Course).

## 👥 Authors

TMG Project Team - Fall 2025

## 🙏 Acknowledgments

- FPT University - Text Mining Course
- Vietnamese NLP community
- Open source libraries: pandas, scikit-learn, DVC, W&B

---

**Built with ❤️ for Vietnamese Text Mining**