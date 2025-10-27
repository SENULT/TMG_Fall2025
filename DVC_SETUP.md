# DVC Setup Instructions

## Prerequisites
- Git initialized
- Python environment activated
- DVC installed (`pip install dvc`)

## Setup Steps

### 1. Initialize DVC
```bash
dvc init
```

### 2. Configure Remote Storage (Optional)

#### For Local Remote
```bash
dvc remote add -d myremote /path/to/local/storage
```

#### For S3
```bash
dvc remote add -d myremote s3://mybucket/path
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY
```

#### For Google Cloud Storage
```bash
dvc remote add -d myremote gs://mybucket/path
```

#### For Azure Blob Storage
```bash
dvc remote add -d myremote azure://mycontainer/path
dvc remote modify myremote account_name YOUR_ACCOUNT_NAME
dvc remote modify myremote account_key YOUR_ACCOUNT_KEY
```

### 3. Track Data with DVC

#### Track raw data
```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw data with DVC"
```

#### Track processed data
```bash
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "Track processed data with DVC"
```

#### Track models (when available)
```bash
dvc add models
git add models.dvc .gitignore
git commit -m "Track models with DVC"
```

### 4. Push Data to Remote
```bash
dvc push
```

### 5. Pull Data from Remote (on another machine)
```bash
dvc pull
```

## DVC Pipeline (Optional - for reproducibility)

Create a `dvc.yaml` file to define your data pipeline:

```yaml
stages:
  preprocess:
    cmd: python src/data_pipeline.py
    deps:
      - data/raw
      - src/data_pipeline.py
      - src/preprocess.py
      - configs/config.yaml
    outs:
      - data/processed
      - splits
    metrics:
      - data/processed/statistics.json:
          cache: false
```

Run the pipeline:
```bash
dvc repro
```

## Common DVC Commands

- `dvc status` - Check DVC status
- `dvc diff` - Show changes in DVC-tracked files
- `dvc checkout` - Restore DVC-tracked files
- `dvc pull` - Download data from remote storage
- `dvc push` - Upload data to remote storage

## Notes

- DVC files (*.dvc) should be committed to Git
- Actual data files are listed in .gitignore
- Use `dvc gc` to clean up old versions of data
- Use `dvc metrics show` to view metrics
