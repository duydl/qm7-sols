import torch
import pytorch_lightning as pl

def create_pl_trainer(args):
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
        dirpath=f"{args.log_dir}/{args.model}/f{args.fold}_{args.version}",
        filename='{epoch:02d}-{train_mae:.2f}-{val_mae:.2f}',
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
    return trainer