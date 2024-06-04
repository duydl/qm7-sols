import os, argparse, shutil

import torch
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import networkx as nx
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader as pyg_loader
from models import *

def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    train_loader = pyg_loader(train_dataset, batch_size=batch_size, num_workers=12)
    val_loader = pyg_loader(val_dataset, batch_size=batch_size, num_workers=12)
    return train_loader, val_loader

def main(args):
    # Load dataset
    data_path = args.data_path
    processed_path = os.path.join(data_path, 'processed')
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    data_train = QM7(data_path, fold=args.fold, train=True)
    data_val = QM7(data_path, fold=args.fold, train=False)
    
    # Create model
    if args.model == "CustomGNN_CoulombGroup":
        # Process dataset
        data_train.transform = T.Compose([CoulombGroupTransform(k=10)])
        data_val.transform = T.Compose([CoulombGroupTransform(k=10)])
        node_features = data_train[0].x.shape[1]
        # edge_attr not yet supported
        # edge_dim = data_train[0].edge_attr.shape[1]
        edge_dim = 0
        model = CustomGNN_1(node_features=node_features, edge_dim=edge_dim)
        
    elif args.model == "CustomGNN_1":
        # Process dataset
        data_train.transform = T.Compose([PruneZeroCharge(), KNNGroupWithPos(k=5)])
        data_val.transform = T.Compose([PruneZeroCharge(), KNNGroupWithPos(k=5)])
        node_features = data_train[0].x.shape[1]
        model = CustomGNN_1(node_features=node_features)
        
    elif args.model == "CustomGNN_2":
        # Process dataset
        data_train.transform = T.Compose([PruneZeroCharge(), KNNGroupWithPos(k=5)])
        data_val.transform = T.Compose([PruneZeroCharge(), KNNGroupWithPos(k=5)])
        node_features = data_train[0].x.shape[1]
        model = CustomGNN_2(node_features=node_features)
        
    elif args.model == "CustomGNN_3":
        # Process dataset
        data_train.transform = T.Compose([PruneZeroCharge(), KNNGroupWithPos(k=5)])
        data_val.transform = T.Compose([PruneZeroCharge(), KNNGroupWithPos(k=5)])
        node_features = data_train[0].x.shape[1]
        model = CustomGNN_3(node_features=node_features)
        
    elif args.model == "GATNet":
        # Process dataset
        data_train.transform = T.Compose([ConcatPosToX(), KNNGroupWithPos(k=10)])
        data_val.transform = T.Compose([ConcatPosToX(), KNNGroupWithPos(k=10)])
        node_features = data_train[0].x.shape[1]
        model = GATNet(node_features=node_features)
        
    elif args.model == "DimeNet":
        # Process dataset
        data_train.transform = T.Compose([PruneZeroCharge()])
        data_val.transform = T.Compose([PruneZeroCharge()])
        model = DimeNetModel()
    else:
        raise ValueError(f"Unknown model name: {args.model}")
    
    print(f"Length train / val: {len(data_train)} / {len(data_val)}, Info: {data_train[0]}")
    print(f"Sample...: \n{data_train[0].x[:2]}")
    sample = data_train[10]
    print('Edge attr (Coulomb force):', sample.edge_attr)
    if sample.num_edges:
        print(f'Node degree: {sample.num_edges / sample.num_nodes:.2f}')
        print(f'Has isolated nodes: {sample.has_isolated_nodes()}')
        print(f'Has self-loops: {sample.has_self_loops()}')
        print(f'Is undirected: {sample.is_undirected()}')
        nx.draw(pyg_utils.to_networkx(sample))
        # plt.show()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_train, data_val, batch_size=args.batch_size)
    
    model_pl = GNNPL(model=model, learning_rate=args.learning_rate)

    # Setup loggers
    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=args.model,
        run_name=f"f{args.fold}_{args.version}",
        tracking_uri=args.mlflow_uri,
        )
    csv_logger = pl.loggers.CSVLogger(
        save_dir=args.log_dir, 
        name=args.model, 
        version=f"f{args.fold}_{args.version}",
        )
    loggers= [mlflow_logger, csv_logger]

    # Determine device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create trainer
    summary_callback = pl.callbacks.ModelSummary(max_depth=8)
    pb_callback = pl.callbacks.RichProgressBar()
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='train_mae', 
        patience=12,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_mae',
        dirpath=args.log_dir + '/best_model',
        filename=f"{args.model}_f{args.fold}_{args.version}"+'-{epoch:02d}-{train_mae:.2f}-{val_mae:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    callbacks = [summary_callback, 
                #  pb_callback,
                 early_stopping,
                 checkpoint_callback,
                 ]
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs+1,
        accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
        callbacks=callbacks,
        logger=loggers
    )

    # Train model
    trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model on the QM7 dataset.")
    parser.add_argument('--model', type=str, default='CustomGNN', 
                        help='The model to use for training (default: CustomGNN)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold (default: 0)')
    parser.add_argument('--version', type=int, default=0,
                        help='Model version (default: 0)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for the optimizer (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs for training (default: 100)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    parser.add_argument('--mlflow_uri', type=str, default=os.path.expanduser('~/mlruns'),
                        help='MLflow tracking URI')
    parser.add_argument('--data_path', type=str, default='../../data',
                        help='Data directory (default: ../../data)')
    args = parser.parse_args()
    main(args)