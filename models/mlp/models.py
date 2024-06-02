import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from torchmetrics import MeanAbsoluteError, MeanSquaredError

import logging

# Configure logging
logging.basicConfig(
    filename='debug.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.DEBUG
)

class Input_SortedEigen(nn.Module):
    def __init__(self, dataset):
        super(Input_SortedEigen, self).__init__()
        self.output_size = 23
        sorted_eigenvals = self.get_sorted_eigenvals(dataset)
        self.mean = sorted_eigenvals.mean(dim=0)
        self.std = (sorted_eigenvals - self.mean).std().item()

    def get_sorted_eigenvals(self, coulomb_matrices):
        eigenvals = torch.linalg.eigvalsh(coulomb_matrices)
        
        # Sort the eigenvalues by their absolute values in descending order
        sorted_eigenvals = torch.sort(torch.abs(eigenvals), dim=1, descending=True).values
        return sorted_eigenvals
    
    def forward(self, coulomb_matrices):
        return (self.get_sorted_eigenvals(coulomb_matrices) - self.mean) / self.std
        
class Input_RandomSortecCM(nn.Module):
    def __init__(self, dataset, step=1.0, noise=1.0):
        super(Input_RandomSortecCM, self).__init__()
        self.step = step
        self.noise = noise
        self.triuind = (torch.arange(23)[:, None] <= torch.arange(23)[None, :]).flatten()
        self.device = dataset.device
        self.max = torch.Tensor([0.0]).to(self.device)
        for _ in range(10): self.max = torch.maximum(self.max,self.realize(dataset).max(dim=0)[0])
        print("self max" ,self.max,self.max.shape)
        
        realized_dataset_ = self.realize(dataset)
        print("realized_dataset", realized_dataset_.shape, realized_dataset_)
        realized_dataset = self.expand(realized_dataset_)
        print("realized_dataset", realized_dataset.shape, realized_dataset)
        self.output_size = realized_dataset.shape[1]
        self.mean = realized_dataset.mean(dim=0)
        self.std = (realized_dataset - self.mean).std().item()
        print("self.std",self.std)

    def realize(self, X):
        def _realize_(x):
            inds = torch.argsort(-(x**2).sum(dim=0)**0.5 + torch.normal(0, self.noise, size=(x.size(0),)))
            x = x[inds, :][:, inds] * 1
            x = x.flatten()[self.triuind]
            return x
        return torch.stack([_realize_(z) for z in X])

    def expand(self, X):
        Xexp = []
        # print("X.shape[1]", X.shape[1])
        for i in range(X.shape[1]):
            for k in torch.arange(0, self.max[i], self.step):
                Xexp.append(torch.tanh((X[:, i] - k) / self.step))
        # print(len(Xexp))
        return torch.stack(Xexp).T

    def forward(self, X, noise):
        X_realized = self.realize(X)
        X_expanded = self.expand(X_realized)
        X_normalized = (X_expanded - self.mean) / self.std

        return X_normalized
    
class CustomLinear(nn.Module):
    def __init__(self, m, n):
        super(CustomLinear, self).__init__()
        self.tr = m**0.5 / n**0.5
        self.lr = 1 / m**0.5

        # Parameters
        self.W = nn.Parameter(torch.tensor(torch.normal(0, 1 / m**0.5, [m, n])))
        self.A = nn.Parameter(torch.zeros(m, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(n, dtype=torch.float32))
    
    def forward(self, X):
        self.X = X
        Y = torch.matmul(X - self.A, self.W) + self.B
        return Y
    
    # def backward(self, DY):
    #     self.DW = torch.matmul((self.X - self.A).T, DY)
    #     self.DA = -(self.X - self.A).sum(dim=0)
    #     self.DB = DY.sum(dim=0) + torch.matmul(self.DA, self.W)
    #     DX = self.tr * torch.matmul(DY, self.W.T)
    #     return DX
    
    # def update(self, lr):
    #     with torch.no_grad():
    #         self.W -= lr * self.lr * self.DW
    #         self.B -= lr * self.lr * self.DB
    #         self.A -= lr * self.lr * self.DA

class Output(nn.Module):
    def __init__(self, T=None):
        super(Output, self).__init__()
        if T is not None:
            self.tmean = T.mean().item()
            self.tstd = T.std().item()
        else:
            self.tmean = 0
            self.tstd = 1
        self.input_size = 1

    def forward(self, X):
        # self.X = X.flatten()
        self.X = X
        return self.X * self.tstd + self.tmean

    # def backward(self, DY):
    #     return (DY / self.tstd).view(-1, 1).type(torch.float32)

class MLP(nn.Module):
    def __init__(self, preprocessor, postprocessor, hidden_sizes=[400, 100], activation_type="sigmoid"):
        super(MLP, self).__init__()
        
        self.preprocess = preprocessor
        self.postprocess = postprocessor
        input_size = self.preprocess.output_size
        output_size = self.postprocess.input_size
        
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

    def forward(self, X):
        processed_X = self.preprocess(X)
        out = self.network(processed_X)
        return self.postprocess(out)


class ModelPL(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, batch_size=64):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.criterion = torch.nn.L1Loss()
        
        # Regression metrics
        # self.train_mae = MeanAbsoluteError()
        # self.val_mae = MeanAbsoluteError()
        # self.test_mae = MeanAbsoluteError()
        
        # self.train_mse = MeanSquaredError()
        # self.val_mse = MeanSquaredError()
        # self.test_mse = MeanSquaredError()

    def forward(self, data):
        return self.model(data)
    
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
        # return []

    def training_step(self, data, batch_idx):
        inputs, targets = data[0], data[1]
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        # self.train_mae(outputs, targets)
        # self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        # self.train_mse(outputs, targets)
        # self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        return loss

    def validation_step(self, data, batch_idx):
        inputs, targets = data[0], data[1]
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        # self.val_mse(outputs, targets)
        # self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        return loss
