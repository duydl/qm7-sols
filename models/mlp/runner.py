import scipy
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import MLP, ModelPL, Input_RandomSortedCM, Input_SortedEigen, Output

import logging

# Configure logging
logging.basicConfig(
    filename='debug.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.DEBUG
)

def load_data(filepath, fold=None):
    dataset = scipy.io.loadmat(filepath)
    if fold == None:
        return dataset['X'], dataset['T'].squeeze()
    else:
        ids_train = dataset['P'][list(range(0, fold)) + list(range(fold+1, 5))].flatten()
        ids_test = dataset['P'][list(range(fold, fold+1))].flatten()
        
        X_train = coulomb_matrices = dataset['X'][ids_train]
        y_train = atom_es = dataset['T'][0, ids_train]
        
        X_test = coulomb_matrices = dataset['X'][ids_test]
        y_test = atom_es = dataset['T'][0, ids_test]

        return X_train, y_train, X_test, y_test

def create_data_loader(X, y, batch_size=32):
    X_tensor = torch.from_numpy(X.copy())
    y_tensor = torch.from_numpy(y.copy()).unsqueeze(-1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return data_loader

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    torch.set_float32_matmul_precision('medium')
    
    filepath = "../../data/raw/qm7.mat"

    X_train, y_train, X_test, y_test = load_data(filepath, fold=0)

    output_size = 1

    # Initialize the MLP and ModelP
    preprocessor = Input_RandomSortedCM(torch.from_numpy(X_train)
                                # .to('cuda' if torch.cuda.is_available() else 'cpu')
                                .to('cpu')
                                )

    # preprocessor = Input_SortedEigen(torch.from_numpy(X_train)
    #                             # .to('cuda' if torch.cuda.is_available() else 'cpu')
    #                             .to('cpu')
    #                             )
    postprocessor = Output(torch.from_numpy(y_train))
    mlp = MLP(preprocessor=preprocessor, postprocessor=postprocessor, activation_type="relu")
    mlp_pl = ModelPL(model=mlp, learning_rate=0.001, batch_size=64)

    # Create dataloaders
    train_loader = create_data_loader(X_train, y_train)
    val_loader = create_data_loader(X_test, y_test)

    # Initialize PyTorch Lightning logger and callbacks
    logger = pl.loggers.CSVLogger(save_dir='logs', 
                                  name='model_training', 
                                  version=0,
                                  )
    callbacks = []
    
    # Configure and run the trainer
    trainer = pl.Trainer(max_epochs=250, 
                        #  accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         accelerator='cpu',
                         callbacks=callbacks,
                         logger=logger,
                         )

    # Fit the model
    trainer.fit(mlp_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)