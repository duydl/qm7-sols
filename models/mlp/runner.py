import scipy
import torch
import sys, os, argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from models import MLP, ModelPL, Input_RandomSortedCM, Input_SortedCM, Input_SortedEigen, Output

sys.path.append("../../utils")
from data import get_cv_fold

def load_data(filepath, fold=None):
    dataset = scipy.io.loadmat(filepath)
    if fold is None:
        return dataset["X"], dataset["T"].squeeze()
    else:
        ids_train = dataset["P"][list(range(0, fold)) + list(range(fold + 1, 5))].flatten()
        ids_test = dataset["P"][list(range(fold, fold + 1))].flatten()
        
        X_train = dataset["X"][ids_train]
        y_train = dataset["T"][0, ids_train]
        
        X_test = dataset["X"][ids_test]
        y_test = dataset["T"][0, ids_test]

        return X_train, y_train, X_test, y_test

def create_data_loader(X, y, batch_size=32):
    X_tensor = torch.from_numpy(X.copy())
    y_tensor = torch.from_numpy(y.copy()).unsqueeze(-1)
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader

def main(args):

    X_train, y_train, X_test, y_test = load_data(args.data_path, fold=args.fold)

    # Initialize the preprocessor
    if args.prep == "random_sorted_cm":
        preprocessor = Input_RandomSortedCM(torch.from_numpy(X_train).to(args.device))
    elif args.prep == "sorted_cm":
        preprocessor = Input_SortedCM(torch.from_numpy(X_train).to(args.device))
    elif args.prep == "sorted_eigen":
        preprocessor = Input_SortedEigen(torch.from_numpy(X_train).to(args.device))
    else:
        raise ValueError(f"Unknown preprocessor type: {args.prep}")

    postprocessor = Output(torch.from_numpy(y_train))
    mlp = MLP(preprocessor=preprocessor, postprocessor=postprocessor, activation_type="relu")
    mlp_pl = ModelPL(model=mlp, learning_rate=args.learning_rate, batch_size=args.batch_size)

    # Create dataloaders
    train_loader = create_data_loader(X_train, y_train, batch_size=args.batch_size)
    val_loader = create_data_loader(X_test, y_test, batch_size=args.batch_size)

    # Setup loggers
    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=f"MLP_{args.prep}",
        run_name=f"f{args.fold}_{args.version}",
        tracking_uri=args.mlflow_uri,
    )
    csv_logger = pl.loggers.CSVLogger(
        save_dir=args.log_dir, 
        name=f"MLP_{args.prep}", 
        version=f"f{args.fold}_{args.version}",
    )
    loggers = [mlflow_logger, csv_logger]

    # Create trainer callbacks
    summary_callback = pl.callbacks.ModelSummary(max_depth=8)
    # pb_callback = pl.callbacks.RichProgressBar()
    pb_callback = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="train_mae", 
        patience=5,
        verbose=False,
        mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_mae",
        dirpath=args.log_dir + "/best_model",
        filename=args.prep + "-MLP-{version}-{epoch:02d}-{train_mae:.2f}-{val_mae:.2f}",
        save_top_k=1,
        verbose=False,
        mode="min",
    )
    
    callbacks = [summary_callback, 
                 early_stopping, 
                 checkpoint_callback,
                 pb_callback,
                ]
    
    # Configure and run the trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs+1, 
        accelerator=args.device,
        callbacks=callbacks,
        logger=loggers,
    )

    # Fit the model
    trainer.fit(mlp_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MLP model on the QM7 dataset.")

    parser.add_argument("--prep", type=str, default="sorted_eigen",
                        choices=["random_sorted_cm", "sorted_cm", "sorted_eigen"], 
                        help="Type of preprocessor to use.")
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold (default: 0)')
    parser.add_argument('--version', type=int, default=0,
                        help='Model version (default: 0)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for the optimizer (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of epochs for training (default: 100)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    parser.add_argument('--mlflow_uri', type=str, default=os.path.expanduser('~/mlruns'),
                        help='MLflow tracking URI')
    parser.add_argument('--data_path', type=str, default='../../data/raw/qm7.mat',
                        help='Data directory (default: ../../data/raw/qm7.mat)')
    parser.add_argument("--device", 
                        type=str,
                        default="cpu",
                        help="Device to use for training (cpu or cuda).")
    
    args = parser.parse_args()
    main(args)
