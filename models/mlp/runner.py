import scipy
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import MLP, ModelPL

def load_data(filepath, fold=1, feature="eigenspectrum"):
    dataset = scipy.io.loadmat(filepath)
    y = atom_es = dataset["T"].squeeze()
    X = coulomb_matrices = dataset["X"]
    return X, y

def create_data_loaders(X, y, batch_size=64, split_ratio=0.8):
    # Convert data to torch tensors
    X_tensor = torch.from_numpy(X.copy())
    y_tensor = torch.from_numpy(y.copy()).unsqueeze(-1)
    
    # Create a dataset from tensors
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split dataset into training and validation sets
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 2)
    
    return train_loader, val_loader

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    torch.set_float32_matmul_precision('medium')
    
    filepath = "../../data/raw/qm7.mat"
    X, y = load_data(filepath)

    matrix_size = X.shape[1]
    output_size = 1

    # Initialize the MLP and ModelPL
    mlp = MLP(matrix_size=matrix_size, output_size=output_size, activation_type="relu")
    mlp_pl = ModelPL(model=mlp, learning_rate=0.01, batch_size=64)

    # Create dataloaders
    train_loader, val_loader = create_data_loaders(X, y)

    # Initialize PyTorch Lightning logger and callbacks
    logger = pl.loggers.CSVLogger(save_dir='logs', 
                                  name='model_training', 
                                  version=0)
    callbacks = []
    
    # Configure and run the trainer
    trainer = pl.Trainer(max_epochs=150, 
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         callbacks=callbacks,
                         logger=logger)

    # Fit the model
    trainer.fit(mlp_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)