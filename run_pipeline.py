"""
Main Pipeline Runner
TMG Fall 2025 - Text Mining & Generation

Run complete data preprocessing pipeline with tracking
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline import DataProcessor
from tracking import WandBTracker, MLflowTracker


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TMG Data Processing Pipeline')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step'
    )
    
    parser.add_argument(
        '--skip-tracking',
        action='store_true',
        help='Skip experiment tracking'
    )
    
    parser.add_argument(
        '--run-eda',
        action='store_true',
        help='Run EDA after preprocessing'
    )
    
    return parser.parse_args()


def run_eda():
    """Run EDA script"""
    print("\n" + "=" * 80)
    print("Running EDA...")
    print("=" * 80)
    
    try:
        # Change to notebooks directory and run EDA
        os.system('cd notebooks && python EDA.py')
        print("\n✓ EDA completed successfully")
    except Exception as e:
        print(f"\n⚠️ EDA failed: {e}")


def main():
    """Main pipeline runner"""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("TMG FALL 2025 - DATA PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print("=" * 80 + "\n")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize trackers
    wandb_tracker = None
    mlflow_tracker = None
    
    if not args.skip_tracking:
        wandb_tracker = WandBTracker(config)
        mlflow_tracker = MLflowTracker(config)
        
        # Start tracking
        run_name = f"preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_tracker.init_run(run_name=run_name, job_type="preprocessing")
        mlflow_tracker.start_run(run_name=run_name)
    
    # Run preprocessing pipeline
    if not args.skip_preprocessing:
        print("\n" + "=" * 80)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 80 + "\n")
        
        processor = DataProcessor(args.config)
        train_df, val_df, test_df = processor.run_pipeline()
        
        # Log to trackers
        if not args.skip_tracking:
            # Load statistics
            stats_path = Path(config['paths']['processed_data']) / 'statistics.json'
            if stats_path.exists():
                import json
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # Log to W&B
                if wandb_tracker and wandb_tracker.enabled:
                    wandb_tracker.log_preprocessing_stats(stats)
                    
                    # Log artifacts
                    processed_path = Path(config['paths']['processed_data'])
                    wandb_tracker.log_data_artifacts(
                        str(processed_path / 'train.parquet'),
                        str(processed_path / 'val.parquet'),
                        str(processed_path / 'test.parquet'),
                        str(Path(config['paths']['splits']) / 'splits.json')
                    )
                
                # Log to MLflow
                if mlflow_tracker and mlflow_tracker.enabled:
                    # Flatten stats for MLflow
                    metrics = {}
                    for split_name, split_stats in stats.items():
                        metrics[f"{split_name}_num_samples"] = split_stats['num_samples']
                        metrics[f"{split_name}_avg_text_length"] = split_stats['avg_text_length']
                        metrics[f"{split_name}_avg_num_words"] = split_stats['avg_num_words']
                    
                    mlflow_tracker.log_metrics(metrics)
                    mlflow_tracker.log_artifacts(str(Path(config['paths']['processed_data'])))
        
        print("\n✓ Preprocessing completed successfully")
    
    # Run EDA
    if args.run_eda:
        run_eda()
        
        # Log EDA figures to W&B
        if not args.skip_tracking and wandb_tracker and wandb_tracker.enabled:
            figures_dir = Path('notebooks/figures')
            if figures_dir.exists():
                wandb_tracker.log_figures(str(figures_dir))
    
    # Finish tracking
    if not args.skip_tracking:
        if wandb_tracker:
            wandb_tracker.finish()
        if mlflow_tracker:
            mlflow_tracker.end_run()
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutput files:")
    print("  - data/processed/train.parquet")
    print("  - data/processed/val.parquet")
    print("  - data/processed/test.parquet")
    print("  - splits/splits.json")
    print("  - data/processed/statistics.json")
    
    if args.run_eda:
        print("  - notebooks/figures/*.png")
        print("  - notebooks/figures/eda_summary.json")
    
    print("\nNext steps:")
    print("  1. Review EDA results in notebooks/figures/")
    print("  2. Setup DVC: see DVC_SETUP.md")
    print("  3. Review data card: DATA_CARD.md")
    print("  4. Start model training!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
