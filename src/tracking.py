"""
Weights & Biases (W&B) Integration
TMG Fall 2025

Setup and tracking for data preprocessing experiments
"""

import wandb
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


class WandBTracker:
    """Weights & Biases experiment tracker"""
    
    def __init__(self, config: Dict):
        """Initialize W&B tracker"""
        self.config = config
        self.tracking_config = config.get('tracking', {})
        self.wandb_config = self.tracking_config.get('wandb', {})
        
        self.enabled = self.tracking_config.get('use_wandb', False)
        self.run = None
    
    def init_run(self, run_name: Optional[str] = None, job_type: str = "preprocessing"):
        """
        Initialize W&B run
        
        Args:
            run_name: Name for this run
            job_type: Type of job (preprocessing, training, etc.)
        """
        if not self.enabled:
            print("W&B tracking disabled")
            return
        
        try:
            self.run = wandb.init(
                project=self.wandb_config.get('project', 'tmg-fall2025'),
                entity=self.wandb_config.get('entity', None),
                config=self.config,
                name=run_name,
                job_type=job_type,
                tags=self.wandb_config.get('tags', []),
                reinit=True
            )
            
            print(f"✓ W&B run initialized: {self.run.name}")
            print(f"  View at: {self.run.url}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize W&B: {e}")
            self.enabled = False
    
    def log_preprocessing_stats(self, stats: Dict):
        """Log preprocessing statistics"""
        if not self.enabled or not self.run:
            return
        
        try:
            # Log statistics for each split
            for split_name, split_stats in stats.items():
                wandb.log({
                    f"{split_name}/num_samples": split_stats.get('num_samples', 0),
                    f"{split_name}/avg_text_length": split_stats.get('avg_text_length', 0),
                    f"{split_name}/avg_num_words": split_stats.get('avg_num_words', 0),
                    f"{split_name}/avg_num_sentences": split_stats.get('avg_num_sentences', 0),
                })
                
                # Log sentiment distribution
                for sentiment, count in split_stats.get('sentiment_distribution', {}).items():
                    wandb.log({f"{split_name}/sentiment_{sentiment}": count})
                
                # Log topic distribution
                for topic, count in split_stats.get('topic_distribution', {}).items():
                    wandb.log({f"{split_name}/topic_{topic}": count})
            
            print("✓ Logged preprocessing stats to W&B")
            
        except Exception as e:
            print(f"⚠️ Failed to log stats: {e}")
    
    def log_data_artifacts(self, 
                          train_path: str, 
                          val_path: str, 
                          test_path: str,
                          splits_path: str):
        """
        Log data artifacts to W&B
        
        Args:
            train_path: Path to train parquet
            val_path: Path to val parquet
            test_path: Path to test parquet
            splits_path: Path to splits JSON
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name='tmg-processed-data',
                type='dataset',
                description='Preprocessed TMG dataset with stratified splits'
            )
            
            # Add files
            artifact.add_file(train_path, name='train.parquet')
            artifact.add_file(val_path, name='val.parquet')
            artifact.add_file(test_path, name='test.parquet')
            artifact.add_file(splits_path, name='splits.json')
            
            # Log artifact
            self.run.log_artifact(artifact)
            
            print("✓ Logged data artifacts to W&B")
            
        except Exception as e:
            print(f"⚠️ Failed to log artifacts: {e}")
    
    def log_figures(self, figures_dir: str):
        """
        Log EDA figures to W&B
        
        Args:
            figures_dir: Directory containing figure files
        """
        if not self.enabled or not self.run:
            return
        
        try:
            figures_path = Path(figures_dir)
            
            if not figures_path.exists():
                print(f"⚠️ Figures directory not found: {figures_dir}")
                return
            
            # Log all PNG files
            for fig_file in figures_path.glob('*.png'):
                wandb.log({
                    fig_file.stem: wandb.Image(str(fig_file))
                })
            
            print(f"✓ Logged figures from {figures_dir} to W&B")
            
        except Exception as e:
            print(f"⚠️ Failed to log figures: {e}")
    
    def log_class_imbalance(self, imbalance_data: Dict):
        """
        Log class imbalance information
        
        Args:
            imbalance_data: Dictionary with imbalance metrics
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb.log(imbalance_data)
            print("✓ Logged class imbalance data to W&B")
            
        except Exception as e:
            print(f"⚠️ Failed to log class imbalance: {e}")
    
    def finish(self):
        """Finish W&B run"""
        if self.enabled and self.run:
            self.run.finish()
            print("✓ W&B run finished")


class MLflowTracker:
    """MLflow experiment tracker (alternative to W&B)"""
    
    def __init__(self, config: Dict):
        """Initialize MLflow tracker"""
        self.config = config
        self.tracking_config = config.get('tracking', {})
        self.mlflow_config = self.tracking_config.get('mlflow', {})
        
        self.enabled = self.tracking_config.get('use_mlflow', False)
        
        if self.enabled:
            import mlflow
            self.mlflow = mlflow
            
            # Setup tracking URI
            tracking_uri = self.mlflow_config.get('tracking_uri', 'mlruns')
            self.mlflow.set_tracking_uri(tracking_uri)
            
            # Setup experiment
            experiment_name = self.mlflow_config.get('experiment_name', 'TMG_Preprocessing')
            self.mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run"""
        if not self.enabled:
            return
        
        try:
            self.mlflow.start_run(run_name=run_name)
            print(f"✓ MLflow run started")
        except Exception as e:
            print(f"⚠️ Failed to start MLflow run: {e}")
    
    def log_params(self, params: Dict):
        """Log parameters"""
        if not self.enabled:
            return
        
        try:
            self.mlflow.log_params(params)
            print("✓ Logged parameters to MLflow")
        except Exception as e:
            print(f"⚠️ Failed to log params: {e}")
    
    def log_metrics(self, metrics: Dict):
        """Log metrics"""
        if not self.enabled:
            return
        
        try:
            self.mlflow.log_metrics(metrics)
            print("✓ Logged metrics to MLflow")
        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")
    
    def log_artifacts(self, artifacts_dir: str):
        """Log artifacts directory"""
        if not self.enabled:
            return
        
        try:
            self.mlflow.log_artifacts(artifacts_dir)
            print(f"✓ Logged artifacts from {artifacts_dir} to MLflow")
        except Exception as e:
            print(f"⚠️ Failed to log artifacts: {e}")
    
    def end_run(self):
        """End MLflow run"""
        if self.enabled:
            self.mlflow.end_run()
            print("✓ MLflow run ended")


def setup_tracking(config_path: str = 'configs/config.yaml'):
    """
    Setup experiment tracking based on configuration
    
    Returns:
        Tuple of (wandb_tracker, mlflow_tracker)
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    wandb_tracker = WandBTracker(config)
    mlflow_tracker = MLflowTracker(config)
    
    return wandb_tracker, mlflow_tracker


if __name__ == "__main__":
    # Example usage
    wandb_tracker, mlflow_tracker = setup_tracking()
    
    # Initialize W&B
    wandb_tracker.init_run(run_name="preprocessing-test", job_type="preprocessing")
    
    # Log example stats
    stats = {
        'train': {
            'num_samples': 1000,
            'avg_text_length': 250.5,
            'avg_num_words': 45.2
        }
    }
    wandb_tracker.log_preprocessing_stats(stats)
    
    # Finish
    wandb_tracker.finish()
    
    print("\n✓ Tracking setup complete!")
