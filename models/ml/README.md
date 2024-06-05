
## Machine Learning Method for QM7 Dataset

### Dependency

### Usage
```
python runner.py --help
usage: runner.py [-h] [--feature FEATURE] [--models MODELS [MODELS ...]] [--log_dir LOG_DIR]
                 [--mlflow_uri MLFLOW_URI] [--data_path DATA_PATH]

Train and evaluate models on the QM7 dataset.

optional arguments:
  -h, --help            show this help message and exit
  --feature FEATURE
  --models MODELS [MODELS ...]
  --log_dir LOG_DIR
  --mlflow_uri MLFLOW_URI
  --data_path DATA_PATH
```

### Test
```
python -m pytest test.py
```