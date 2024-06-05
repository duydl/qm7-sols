## Graph Neural Networks for QM7 Dataset

### Dependency

### Usage
```
python runner.py --help
usage: runner.py [-h] [--model MODEL] [--fold FOLD] [--version VERSION]
                 [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                 [--scheduler SCHEDULER] [--max_epochs MAX_EPOCHS] [--log_dir LOG_DIR]
                 [--mlflow_uri MLFLOW_URI] [--data_path DATA_PATH]

Train a GNN model on the QM7 dataset.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The model to use for training (default: CustomGNN)
  --fold FOLD           Fold (default: 0)
  --version VERSION     Model version (default: 0)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.01)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 64)
  --scheduler SCHEDULER
                        Scheduler for optimizer (default: exp)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs for training (default: 100)
  --log_dir LOG_DIR     Directory to save logs (default: logs)
  --mlflow_uri MLFLOW_URI
                        MLflow tracking URI
  --data_path DATA_PATH
                        Data directory (default: ../../data)
```

### Demonstrations

See `demo.ipynb`