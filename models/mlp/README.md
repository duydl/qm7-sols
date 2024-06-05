## Multi Layer Perception for QM7 Dataset

### Dependency

### Usage

```
python runner.py --help
usage: runner.py [-h] [--prep {random_sorted_cm,sorted_cm,sorted_eigen}] [--fold FOLD]
                 [--version VERSION] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                 [--max_epochs MAX_EPOCHS] [--log_dir LOG_DIR] [--mlflow_uri MLFLOW_URI]
                 [--data_path DATA_PATH] [--device DEVICE]

Train a MLP model on the QM7 dataset.

optional arguments:
  -h, --help            show this help message and exit
  --prep {random_sorted_cm,sorted_cm,sorted_eigen}
                        Type of preprocessor to use.
  --fold FOLD           Fold (default: 0)
  --version VERSION     Model version (default: 0)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.01)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs for training (default: 100)
  --log_dir LOG_DIR     Directory to save logs (default: logs)
  --mlflow_uri MLFLOW_URI
                        MLflow tracking URI
  --data_path DATA_PATH
                        Data directory (default: ../../data/raw/qm7.mat)
  --device DEVICE       Device to use for training (cpu or cuda).
```

### Demonstration

See `demo.ipynb`. 