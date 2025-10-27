# TMG Dataset - Data Card

## Dataset Description

**Dataset Name:** TMG Fall 2025 - Vietnamese Text Classification Dataset  
**Version:** 1.0.0  
**Last Updated:** October 2025  
**Language:** Vietnamese (vi)

### Overview
This dataset contains Vietnamese text samples for multi-class classification tasks, focusing on sentiment analysis and topic classification. The data has been preprocessed and split for training, validation, and testing.

## Dataset Structure

### Data Splits

| Split | Samples | Ratio |
|-------|---------|-------|
| Train | ~23,000 | 70% |
| Validation | ~3,300 | 15% |
| Test | ~6,600 | 15% |

**Note:** Exact numbers may vary after preprocessing and deduplication.

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `text` | string | Original Vietnamese text |
| `text_processed` | string | Preprocessed text |
| `sentiment` | float | Sentiment label (0.0, 1.0, 2.0, etc.) |
| `classification` | float | Topic/category label |
| `text_length` | int | Length of processed text (characters) |
| `num_words` | int | Number of words in processed text |
| `num_sentences` | int | Number of sentences |
| `sentences` | list | List of individual sentences |
| `original_split` | string | Original data split (train/val/test) |

## Preprocessing Steps

### 1. Text Normalization
- **Lowercase:** All text converted to lowercase
- **Vietnamese Tones:** Preserved (diacritics kept)
- **Whitespace:** Normalized (multiple spaces → single space)

### 2. Special Token Mapping
- **URLs:** Replaced with `<URL>`
- **Numbers:** Replaced with `<NUM>`
- **Emojis:** Replaced with `<EMOJI>`

### 3. Punctuation Normalization
- Consecutive punctuation limited to max 3 characters
- Vietnamese punctuation marks standardized
- Example: `!!!!!!` → `!!!`

### 4. Teencode Handling
- Common Vietnamese teencode/slang replaced with standard forms
- Examples:
  - `k` → `không`
  - `dc` → `được`
  - `vs` → `với`
  - `ntn` → `như thế nào`

### 5. Sentence Splitting
- Texts split into individual sentences for error analysis
- Sentences preserved in `sentences` field
- Sentence separator: `[SEP]`

### 6. Deduplication
- Method: Fuzzy matching
- Threshold: ~95% similarity
- Duplicate texts removed to ensure data quality

### 7. Length Filtering
- **Minimum length:** 10 characters
- **Maximum length:** 5000 characters
- Texts outside this range removed

## Data Distribution

### Sentiment Distribution
- **Class 0.0:** Neutral
- **Class 1.0:** Positive
- **Class 2.0:** Negative
- **Others:** Additional sentiment categories

*Note: Check `notebooks/figures/sentiment_distribution.png` for detailed distribution.*

### Topic Distribution
- Multiple topic categories (classification labels)
- Stratified split ensures balanced representation across topics

*Note: Check `notebooks/figures/topic_distribution.png` for detailed distribution.*

### Class Imbalance
- **Threshold for "minor" classes:** < 5% of total samples
- Classes below threshold marked for special attention during training
- Both sentiment and topic distributions analyzed for imbalance

## Quality Checks

### Robustness Tests
1. **URL Replacement Test:** Verify model handles `<URL>` token
2. **Number Replacement Test:** Verify model handles `<NUM>` token
3. **Punctuation Permutation:** Test model stability with punctuation changes
4. **Small Class Bias Check:** Monitor performance on minority classes

### Multi-Aspect Analysis
- Texts may contain multiple aspects/sentiments
- Cross-tabulation of sentiment × topic performed
- Multi-aspect ratio calculated and reported

## Data Versioning

### DVC (Data Version Control)
- Raw data tracked: `data/raw.dvc`
- Processed data tracked: `data/processed.dvc`
- Remote storage: Configurable (S3, GCS, Azure, local)

### Git
- Code and configurations version controlled
- Split indices saved in `splits/splits.json`
- Reproducible with fixed seed: **42**

### Tracking
- **W&B (Weights & Biases):** Recommended for experiment tracking
- **MLflow:** Alternative tracking system supported
- Preprocessing logs: `logs/preprocessing.log`

## Usage

### Loading Data

#### Parquet (Recommended)
```python
import pandas as pd

train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')
```

#### CSV (If converted)
```python
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')
```

### Loading Split Indices
```python
import json

with open('splits/splits.json', 'r') as f:
    splits = json.load(f)

train_indices = splits['train_indices']
val_indices = splits['val_indices']
test_indices = splits['test_indices']
```

### Using Configuration
```python
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access preprocessing settings
preprocess_config = config['preprocessing']
split_config = config['splitting']
```

## Limitations

1. **Language-Specific:** Designed for Vietnamese text only
2. **Domain-Specific:** May not generalize to all Vietnamese text domains
3. **Deduplication:** Aggressive deduplication may remove legitimate similar texts
4. **Teencode Coverage:** Teencode dictionary not exhaustive
5. **Sentence Splitting:** May not be perfect for complex Vietnamese sentences
6. **Class Imbalance:** Some classes significantly underrepresented

## Ethical Considerations

- Data should be used responsibly
- Consider potential biases in sentiment and topic labels
- Minority class performance should be monitored
- Privacy: Ensure no personal information in text samples

## Citation

If you use this dataset, please cite:

```
@dataset{tmg_fall2025,
  title={TMG Fall 2025 - Vietnamese Text Classification Dataset},
  author={TMG Project Team},
  year={2025},
  version={1.0.0}
}
```

## Contact & Support

For questions or issues:
- Check documentation in repository
- Review EDA results in `notebooks/figures/`
- See configuration in `configs/config.yaml`

## Changelog

### Version 1.0.0 (October 2025)
- Initial release
- Preprocessing pipeline implemented
- Stratified split (70/15/15)
- DVC setup
- EDA completed
- Documentation finalized
