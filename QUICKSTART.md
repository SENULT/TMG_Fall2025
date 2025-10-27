# Quick Setup Guide - TMG Fall 2025

## 🚀 Quick Start (5 minutes)

### Step 1: Setup Environment

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Test Preprocessing

```powershell
# Test if preprocessing works
python test_preprocessing.py
```

Expected output: ✅ ALL TESTS COMPLETED SUCCESSFULLY!

### Step 3: Run Full Pipeline

```powershell
# Run complete pipeline with EDA
python run_pipeline.py --run-eda
```

This will:
- ✅ Load data from `data/raw/`
- ✅ Preprocess Vietnamese text
- ✅ Create stratified splits (70/15/15)
- ✅ Save to `data/processed/`
- ✅ Generate EDA visualizations

### Step 4: View Results

Check outputs:
- `data/processed/` - Processed parquet files
- `splits/` - Split indices
- `notebooks/figures/` - EDA visualizations
- `logs/` - Processing logs

## 📋 What You Get

### Processed Data
```
data/processed/
├── train.parquet      # ~23,000 samples
├── val.parquet        # ~3,300 samples
├── test.parquet       # ~6,600 samples
└── statistics.json    # Dataset statistics
```

### EDA Visualizations
```
notebooks/figures/
├── sentiment_distribution.png
├── topic_distribution.png
├── text_length_analysis.png
├── sentiment_by_topic.png
└── eda_summary.json
```

### Configuration
- `configs/config.yaml` - Main settings
- `configs/teencode_dict.json` - Vietnamese slang dictionary

## 🔧 Common Issues

### Issue: Module not found

**Solution:**
```powershell
# Make sure you're in project root
cd "d:\fpt university\majority\study\kì 5\Text Mining\TMG_Fall2025"

# Activate venv
.\venv\Scripts\Activate.ps1
```

### Issue: CSV files not found

**Solution:**
```powershell
# Copy original data to data/raw/
Copy-Item "TMG_Dataset\*.csv" -Destination "data\raw\"
```

### Issue: Memory error during deduplication

**Solution:** Edit `configs/config.yaml`
```yaml
preprocessing:
  dedup: false  # Disable deduplication
  # OR
  dedup_method: "exact"  # Use faster exact matching
```

## 🎯 Next Steps

### 1. Review EDA Results
```powershell
cd notebooks/figures
# View PNG files in your image viewer
```

### 2. Setup Version Control (Optional)

#### Git
```powershell
git init
git add .
git commit -m "Initial commit"
```

#### DVC (for data versioning)
```powershell
pip install dvc
dvc init
dvc add data/raw
dvc add data/processed
git add data/*.dvc .gitignore
git commit -m "Track data with DVC"
```

### 3. Setup Experiment Tracking (Optional)

#### Weights & Biases
```powershell
pip install wandb
wandb login
```

Edit `configs/config.yaml`:
```yaml
tracking:
  use_wandb: true
  wandb:
    project: "tmg-fall2025"
    entity: "your-username"  # Your W&B username
```

Run with tracking:
```powershell
python run_pipeline.py --run-eda
```

### 4. Customize Preprocessing

Edit `configs/config.yaml` to customize:
- Text normalization settings
- Special token mapping
- Deduplication threshold
- Split ratios
- Class imbalance threshold

### 5. Load and Use Data

```python
import pandas as pd

# Load processed data
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')

# View sample
print(train[['text_processed', 'sentiment', 'classification']].head())

# Check statistics
print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")
```

## 📚 Documentation

- **README.md** - Full project documentation
- **DATA_CARD.md** - Dataset description
- **DVC_SETUP.md** - Version control guide

## 🆘 Need Help?

1. Check logs: `logs/preprocessing.log`
2. Read full README: `README.md`
3. Review configuration: `configs/config.yaml`
4. Test preprocessing: `python test_preprocessing.py`

## ✅ Verification Checklist

Before proceeding to model training:

- [ ] All dependencies installed
- [ ] Test preprocessing passes
- [ ] Pipeline runs successfully
- [ ] Processed files exist in `data/processed/`
- [ ] EDA figures generated
- [ ] No critical errors in logs
- [ ] Data splits are balanced
- [ ] Statistics look reasonable

If all checked ✅, you're ready for model training!

## 🎓 For FPT Students

Make sure you have:
1. Completed data preprocessing ✅
2. Reviewed EDA results ✅
3. Documented findings ✅
4. Prepared for model training ✅

Good luck with your TMG project! 🚀
