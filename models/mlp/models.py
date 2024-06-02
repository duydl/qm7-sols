import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from torchmetrics import MeanAbsoluteError, MeanSquaredError

class RandomSortCM(nn.Module):
    def __init__(self, matrix_size):
        super(RandomSortCM, self).__init__()
        self.triuind = torch.triu_indices(matrix_size, matrix_size, offset=0).t().contiguous().view(-1)

    def realize(self, X, noise):
        X_flat = []
        for x in X:
            norms = torch.norm(x, dim=1)
            inds = torch.argsort(-norms + torch.randn_like(norms) * noise)
            x_sorted = x[inds, :][:, inds]
            x_flattened = x_sorted.flatten()[self.triuind]
            X_flat.append(x_flattened)
        return torch.stack(X_flat)

    def forward(self, X, noise):
        return self.realize(X, noise)
    

class MLP(nn.Module):
    def __init__(self, matrix_size, output_size, hidden_sizes=[400, 100], activation_type="tanh"):
        super(MLP, self).__init__()
        
        self.preprocess = RandomSortCM(matrix_size)
        input_size = self.preprocess.triuind.size(0)
        
        # Create a list of all sizes: input, hidden layers, output
        all_sizes = [input_size] + hidden_sizes + [output_size]
        
        layers = []
        for i in range(len(all_sizes) - 1):
            layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
            if i < len(all_sizes) - 2:  # Don't add activation after last layer
                if activation_type == "tanh":
                    layers.append(nn.Tanh())
                elif activation_type == "relu":
                    layers.append(nn.ReLU())
                elif activation_type == "sigmoid":
                    layers.append(nn.Sigmoid())
        
        # Unwrap into nn.Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, X, noise=1.0):
        processed_X = self.preprocess(X, noise)
        return self.network(processed_X)


class ModelPL(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, batch_size=64):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.criterion = torch.nn.L1Loss()
        
        # Regression metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, data, noise=1.0):
        return self.model(data, noise=noise)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.25, 
                patience=5),
            'monitor': 'val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        # def lr_lambda(epoch):
        #     if epoch > 12500:
        #         return 0.01
        #     elif epoch > 2500:
        #         return 0.005
        #     elif epoch > 500:
        #         return 0.0025
        #     else:
        #         return 0.001
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, data, batch_idx):
        inputs, targets = data[0], data[1]
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        # self.train_mae(outputs, targets)
        # self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        # self.train_mse(outputs, targets)
        # self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        return loss

    def validation_step(self, data, batch_idx):
        inputs, targets = data[0], data[1]
        outputs = self(inputs, noise=0.0)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        self.val_mse(outputs, targets)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        return loss
