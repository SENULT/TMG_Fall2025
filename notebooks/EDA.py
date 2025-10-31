# EDA - Exploratory Data Analysis
# TMG Fall 2025 - Text Mining & Generation

## Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figures directory
figures_dir = Path('notebooks/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

print("✓ Setup complete")

## Load Processed Data

# Load processed parquet files
train_df = pd.read_parquet('../data/processed/train.parquet')
val_df = pd.read_parquet('../data/processed/val.parquet')
test_df = pd.read_parquet('../data/processed/test.parquet')

# Combine for overall analysis
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")
print(f"Total: {len(all_data)} samples")

## Basic Statistics

# Display sample
print("\n=== Sample Data ===")
print(all_data[['text_processed', 'sentiment', 'classification', 'text_length', 'num_words']].head())

# Basic info
print("\n=== Data Info ===")
print(all_data.info())

# Summary statistics
print("\n=== Summary Statistics ===")
print(all_data[['text_length', 'num_words', 'num_sentences']].describe())

## 1. Class Distribution Analysis

### Sentiment Distribution

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall sentiment distribution
sentiment_counts = all_data['sentiment'].value_counts().sort_index()
axes[0].bar(sentiment_counts.index, sentiment_counts.values, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Sentiment', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Sentiment by split
sentiment_by_split = all_data.groupby(['split', 'sentiment']).size().unstack(fill_value=0)
sentiment_by_split.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_xlabel('Split', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Sentiment Distribution by Split', fontsize=14, fontweight='bold')
axes[1].legend(title='Sentiment')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(axis='y', alpha=0.3)

# Sentiment proportions
sentiment_prop = all_data['sentiment'].value_counts(normalize=True)
axes[2].pie(sentiment_prop.values, labels=sentiment_prop.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title('Sentiment Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Sentiment Distribution ===")
print(sentiment_counts)
print("\nProportions:")
print(sentiment_prop)

### Topic/Classification Distribution

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Topic distribution
topic_counts = all_data['classification'].value_counts().sort_index()
axes[0].bar(topic_counts.index, topic_counts.values, color='lightcoral', edgecolor='black')
axes[0].set_xlabel('Topic/Classification', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Topic Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Topic proportions
topic_prop = all_data['classification'].value_counts(normalize=True)
colors = plt.cm.Set3(range(len(topic_prop)))
axes[1].barh(topic_prop.index.astype(str), topic_prop.values, color=colors, edgecolor='black')
axes[1].set_xlabel('Proportion', fontsize=12)
axes[1].set_ylabel('Topic', fontsize=12)
axes[1].set_title('Topic Proportions', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'topic_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Topic Distribution ===")
print(topic_counts)
print("\nProportions:")
print(topic_prop)

## 2. Class Imbalance Detection

# Mark minor classes (< 5%)
threshold = 0.05

print("\n=== Class Imbalance Analysis ===")
print(f"Threshold for minor classes: {threshold * 100}%\n")

# Sentiment imbalance
print("Sentiment Classes:")
for sent, prop in sentiment_prop.items():
    status = "⚠️ MINOR" if prop < threshold else "✓ OK"
    print(f"  Sentiment {sent}: {prop:.2%} {status}")

# Topic imbalance
print("\nTopic Classes:")
for topic, prop in topic_prop.items():
    status = "⚠️ MINOR" if prop < threshold else "✓ OK"
    print(f"  Topic {topic}: {prop:.2%} {status}")

## 3. Text Length Analysis

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Text length distribution
axes[0, 0].hist(all_data['text_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Text Length (characters)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
axes[0, 0].axvline(all_data['text_length'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 0].axvline(all_data['text_length'].median(), color='green', linestyle='--', linewidth=2, label='Median')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Word count distribution
axes[0, 1].hist(all_data['num_words'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Number of Words', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
axes[0, 1].axvline(all_data['num_words'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 1].axvline(all_data['num_words'].median(), color='green', linestyle='--', linewidth=2, label='Median')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Sentence count distribution
axes[1, 0].hist(all_data['num_sentences'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Number of Sentences', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Sentence Count Distribution', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Text length by sentiment
all_data.boxplot(column='text_length', by='sentiment', ax=axes[1, 1])
axes[1, 1].set_xlabel('Sentiment', fontsize=12)
axes[1, 1].set_ylabel('Text Length', fontsize=12)
axes[1, 1].set_title('Text Length by Sentiment', fontsize=14, fontweight='bold')
plt.sca(axes[1, 1])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(figures_dir / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Text Length Statistics ===")
print(f"Mean length: {all_data['text_length'].mean():.2f} characters")
print(f"Median length: {all_data['text_length'].median():.2f} characters")
print(f"Mean words: {all_data['num_words'].mean():.2f}")
print(f"Mean sentences: {all_data['num_sentences'].mean():.2f}")

## 4. Sentiment by Topic Analysis

# Cross-tabulation
sentiment_by_topic = pd.crosstab(all_data['classification'], all_data['sentiment'], normalize='index')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(sentiment_by_topic, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Proportion'})
axes[0].set_xlabel('Sentiment', fontsize=12)
axes[0].set_ylabel('Topic', fontsize=12)
axes[0].set_title('Sentiment Distribution by Topic (Normalized)', fontsize=14, fontweight='bold')

# Stacked bar chart
sentiment_by_topic_counts = pd.crosstab(all_data['classification'], all_data['sentiment'])
sentiment_by_topic_counts.plot(kind='bar', stacked=True, ax=axes[1], width=0.8)
axes[1].set_xlabel('Topic', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Sentiment Counts by Topic', fontsize=14, fontweight='bold')
axes[1].legend(title='Sentiment')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'sentiment_by_topic.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Sentiment by Topic ===")
print(sentiment_by_topic)

## 5. Multi-Aspect Detection

# Check if there are multiple sentiments for same text patterns
# This is a simplified check - in practice, you'd use more sophisticated methods

print("\n=== Multi-Aspect Analysis ===")

# Group by topic and check sentiment variety
topic_sentiment_variety = all_data.groupby('classification')['sentiment'].nunique()
print("\nNumber of unique sentiments per topic:")
print(topic_sentiment_variety)

# Calculate multi-aspect ratio (topics with multiple sentiments)
multi_aspect_topics = (topic_sentiment_variety > 1).sum()
total_topics = len(topic_sentiment_variety)
multi_aspect_ratio = multi_aspect_topics / total_topics

print(f"\nMulti-aspect ratio: {multi_aspect_ratio:.2%}")
print(f"Topics with multiple sentiments: {multi_aspect_topics}/{total_topics}")

## 6. Data Quality Report

print("\n" + "=" * 80)
print("DATA QUALITY REPORT")
print("=" * 80)

# Missing values
print("\n1. Missing Values:")
print(all_data.isnull().sum())

# Duplicate check (exclude list columns)
print(f"\n2. Duplicates (based on text_processed): {all_data.duplicated(subset=['text_processed']).sum()}")

# Text length outliers
q1 = all_data['text_length'].quantile(0.25)
q3 = all_data['text_length'].quantile(0.75)
iqr = q3 - q1
outliers = ((all_data['text_length'] < (q1 - 1.5 * iqr)) | (all_data['text_length'] > (q3 + 1.5 * iqr))).sum()
print(f"\n3. Text Length Outliers: {outliers} ({outliers/len(all_data)*100:.2f}%)")

# Class balance
print("\n4. Class Balance:")
print(f"   - Most common sentiment: {sentiment_counts.idxmax()} ({sentiment_counts.max()} samples)")
print(f"   - Least common sentiment: {sentiment_counts.idxmin()} ({sentiment_counts.min()} samples)")
print(f"   - Imbalance ratio (max/min): {sentiment_counts.max() / sentiment_counts.min():.2f}x")

print("\n" + "=" * 80)

## Save Summary Statistics

summary_stats = {
    'total_samples': len(all_data),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'avg_text_length': float(all_data['text_length'].mean()),
    'avg_num_words': float(all_data['num_words'].mean()),
    'avg_num_sentences': float(all_data['num_sentences'].mean()),
    'sentiment_distribution': sentiment_counts.to_dict(),
    'topic_distribution': topic_counts.to_dict(),
    'minor_sentiment_classes': [int(k) for k, v in sentiment_prop.items() if v < threshold],
    'minor_topic_classes': [float(k) for k, v in topic_prop.items() if v < threshold],
    'multi_aspect_ratio': float(multi_aspect_ratio),
    'duplicates': int(all_data.duplicated().sum()),
    'outliers': int(outliers)
}

with open(figures_dir / 'eda_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

print("\n✓ EDA complete! All figures saved to notebooks/figures/")
