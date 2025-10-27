"""
Data Processing Pipeline
TMG Fall 2025 - Text Mining & Generation

Pipeline:
1. Load raw data (CSV)
2. Preprocess text (Vietnamese-specific)
3. Stratified split (70/15/15) by (topic, sentiment)
4. Save processed data (Parquet)
5. Save split indices (JSON)
6. Log to W&B/MLflow
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

# Import preprocessing module
from preprocess import VietnameseTextPreprocessor, RobustnessChecker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """Initialize data processor with configuration"""
        self.config = self._load_config(config_path)
        self.preprocessor = VietnameseTextPreprocessor(self.config)
        self.robustness_checker = RobustnessChecker(self.config)
        
        # Setup paths
        self.paths = self.config.get('paths', {})
        self.data_config = self.config.get('data', {})
        self.split_config = self.config.get('splitting', {})
        
        # Set random seed
        self.seed = self.config.get('project', {}).get('seed', 42)
        np.random.seed(self.seed)
        
        logger.info(f"DataProcessor initialized with seed={self.seed}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV files
        Combines train, val, test from original split
        """
        logger.info("Loading raw data...")
        
        raw_path = Path(self.paths['raw_data'])
        
        # Load all CSV files
        dfs = []
        for file_key in ['train_file', 'val_file', 'test_file']:
            file_name = self.data_config[file_key]
            file_path = raw_path / file_name
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['original_split'] = file_key.replace('_file', '')
                dfs.append(df)
                logger.info(f"Loaded {file_name}: {len(df)} samples")
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df)} samples")
        
        return combined_df
    
    def check_stratification_feasibility(self, df: pd.DataFrame, 
                                         stratify_cols: List[str]) -> bool:
        """
        Check if stratified splitting is feasible
        Ensure each combination has enough samples
        """
        min_samples = self.split_config.get('min_samples_per_class', 5)
        
        # Group by stratification columns
        grouped = df.groupby(stratify_cols).size()
        
        # Check minimum samples
        too_small = grouped[grouped < min_samples * 3]  # Need at least 3x for 3 splits
        
        if len(too_small) > 0:
            logger.warning(f"Found {len(too_small)} groups with < {min_samples * 3} samples:")
            logger.warning(f"{too_small.to_dict()}")
            return False
        
        logger.info(f"Stratification check passed. All groups have sufficient samples.")
        return True
    
    def stratified_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified split by (topic, sentiment)
        Ratio: 70/15/15
        """
        logger.info("Performing stratified split...")
        
        stratify_cols = self.split_config.get('stratify_by', ['classification', 'sentiment'])
        train_ratio = self.split_config.get('train_ratio', 0.70)
        val_ratio = self.split_config.get('val_ratio', 0.15)
        test_ratio = self.split_config.get('test_ratio', 0.15)
        
        # Create stratification column
        df['_stratify'] = df[stratify_cols].astype(str).agg('_'.join, axis=1)
        
        # Check feasibility
        if not self.check_stratification_feasibility(df, stratify_cols):
            logger.warning("Stratification may not be perfect due to small groups")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df['_stratify'],
            random_state=self.seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['_stratify'],
            random_state=self.seed
        )
        
        # Remove temporary column
        train_df = train_df.drop(columns=['_stratify'])
        val_df = val_df.drop(columns=['_stratify'])
        test_df = test_df.drop(columns=['_stratify'])
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Verify split ratios
        total = len(df)
        logger.info(f"Split ratios - Train: {len(train_df)/total:.2%}, Val: {len(val_df)/total:.2%}, Test: {len(test_df)/total:.2%}")
        
        return train_df, val_df, test_df
    
    def save_split_indices(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, original_df: pd.DataFrame):
        """
        Save split indices to JSON files
        """
        logger.info("Saving split indices...")
        
        splits_path = Path(self.paths['splits'])
        splits_path.mkdir(parents=True, exist_ok=True)
        
        # Get indices
        train_indices = train_df.index.tolist()
        val_indices = val_df.index.tolist()
        test_indices = test_df.index.tolist()
        
        # Save indices
        splits_data = {
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices)
        }
        
        with open(splits_path / 'splits.json', 'w', encoding='utf-8') as f:
            json.dump(splits_data, f, indent=2)
        
        # Save individual split files
        for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            with open(splits_path / f'{split_name}_indices.json', 'w', encoding='utf-8') as f:
                json.dump(indices, f)
        
        logger.info(f"Saved split indices to {splits_path}")
    
    def compute_statistics(self, df: pd.DataFrame, split_name: str) -> Dict:
        """Compute statistics for a dataset split"""
        stats = {
            'split': split_name,
            'num_samples': len(df),
            'avg_text_length': df['text_length'].mean(),
            'avg_num_words': df['num_words'].mean(),
            'avg_num_sentences': df['num_sentences'].mean(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'topic_distribution': df['classification'].value_counts().to_dict(),
        }
        
        return stats
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame):
        """
        Save processed data to Parquet files
        """
        logger.info("Saving processed data...")
        
        processed_path = Path(self.paths['processed_data'])
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        train_df.to_parquet(processed_path / 'train.parquet', index=False)
        val_df.to_parquet(processed_path / 'val.parquet', index=False)
        test_df.to_parquet(processed_path / 'test.parquet', index=False)
        
        logger.info(f"Saved processed data to {processed_path}")
        
        # Save statistics
        stats = {
            'train': self.compute_statistics(train_df, 'train'),
            'val': self.compute_statistics(val_df, 'val'),
            'test': self.compute_statistics(test_df, 'test')
        }
        
        with open(processed_path / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved statistics")
    
    def run_pipeline(self):
        """
        Run full data processing pipeline
        """
        logger.info("=" * 80)
        logger.info("Starting Data Processing Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Load raw data
        df = self.load_raw_data()
        
        # Step 2: Preprocess
        logger.info("\nStep 2: Preprocessing texts...")
        df = self.preprocessor.preprocess_dataframe(df, text_column='text')
        
        # Step 3: Stratified split
        logger.info("\nStep 3: Performing stratified split...")
        train_df, val_df, test_df = self.stratified_split(df)
        
        # Step 4: Save split indices
        logger.info("\nStep 4: Saving split indices...")
        self.save_split_indices(train_df, val_df, test_df, df)
        
        # Step 5: Save processed data
        logger.info("\nStep 5: Saving processed data...")
        self.save_processed_data(train_df, val_df, test_df)
        
        # Step 6: Check class imbalance
        logger.info("\nStep 6: Checking class imbalance...")
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            logger.info(f"\n{split_name.upper()} set:")
            
            # Sentiment distribution
            sent_dist = self.robustness_checker.check_class_distribution(split_df, 'sentiment')
            logger.info(f"Sentiment distribution: {sent_dist['distribution']}")
            if sent_dist['num_small_classes'] > 0:
                logger.warning(f"Minor sentiment classes: {sent_dist['small_classes']}")
            
            # Topic distribution
            topic_dist = self.robustness_checker.check_class_distribution(split_df, 'classification')
            logger.info(f"Topic distribution: {topic_dist['distribution']}")
            if topic_dist['num_small_classes'] > 0:
                logger.warning(f"Minor topic classes: {topic_dist['small_classes']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Data Processing Pipeline Complete!")
        logger.info("=" * 80)
        
        return train_df, val_df, test_df


def main():
    """Main entry point"""
    processor = DataProcessor()
    train_df, val_df, test_df = processor.run_pipeline()
    
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    print("=" * 80)


if __name__ == "__main__":
    # Create logs directory if not exists
    os.makedirs('logs', exist_ok=True)
    
    # Run pipeline
    main()
