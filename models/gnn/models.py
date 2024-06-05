import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.transforms as T

import networkx as nx
import pytorch_lightning as pl
import scipy
import sys

sys.path.append("../../utils")
from data import get_cv_fold

## GNN Models
# Example of passing edge_attr: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gmm_conv.html#GMMConv
# Could not pass attr to propagate: https://github.com/pyg-team/pytorch_geometric/issues/9059#issuecomment-2143304617
class CustomConvLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, pos_dim=3):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 
                    #   pos_dim, 
                    #   pos_dim * 2, 
                      1,
                      out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, pos, edge_index):
        # Propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, x_i, pos_j, pos_i):
        
        distance = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)
        # edge_feat = torch.cat([x_i, x_j, pos_j - pos_i], dim=-1) # thus 2*in_channels + pos_dim

        # edge_feat = torch.cat([x_i, x_j, pos_i, pos_j], dim=-1)
        edge_feat = torch.cat([x_i, x_j, distance], dim=-1)
        return self.mlp(edge_feat)
    
class CustomConvLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, pos_dim=3, edge_dim=0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.pos_dim = pos_dim
        self.edge_dim = edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + pos_dim + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, edge_index, x, pos, 
                # edge_attr=None
                ):
        # if edge_attr is None:
        #     assert self.edge_dim == 0, f"edge_dim must be 0 when edge_attr is None"
        # else:
        #     assert self.edge_dim == edge_attr.size(1), f"edge_dim {self.edge_dim} must match the size of edge_attr features {edge_attr.size(1)}"

        # Propagating messages.
        return self.propagate(edge_index=edge_index, x=x, pos=pos, 
                            #   edge_attr=edge_attr,
                              )

    def message(self, x_j, x_i, pos_j, pos_i, 
                # edge_attr
                ):
        # if edge_attr is None:
        #     edge_feat = torch.cat([x_i, x_j, pos_j - pos_i], dim=-1)
        # else:
        #     edge_feat = torch.cat([x_i, x_j, pos_j - pos_i, edge_attr], dim=-1)
        if self.pos_dim == 1:
            distance = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)
            edge_feat = torch.cat([x_i, x_j, distance], dim=-1)
        else:
            edge_feat = torch.cat([x_i, x_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)

class CustomGNN_1(nn.Module):
    def __init__(self, node_features=1, pos_dim=3, edge_dim=0, hidden_dim=128, output_dim=1):
        super().__init__()

        self.conv1 = CustomConvLayer(node_features, hidden_dim, pos_dim=pos_dim, edge_dim=edge_dim)
        self.conv2 = CustomConvLayer(hidden_dim, hidden_dim, pos_dim=pos_dim, edge_dim=edge_dim)

        self.predictor = pyg_nn.MLP([hidden_dim, hidden_dim, output_dim], bias=[False, True])

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        
        # First ConvLayer layer
        x = self.conv1(edge_index=edge_index, x=x, pos=pos, 
                    #    edge_attr=edge_attr
                       )
        x = x.relu()
        
        # Second ConvLayer layer
        x = self.conv2(edge_index, x=x, pos=pos, 
                    #    edge_attr=edge_attr
                       )
        x = x.relu()
        
        # Global Pooling:
        x = pyg_nn.global_add_pool(x, batch)
        
        # Predictor
        return self.predictor(x)

class CustomGNN_2(nn.Module):
    def __init__(self, node_features=1, pos_dim=3, edge_dim=0, hidden_dim=128, output_dim=1, num_filters=8):
        super().__init__()

        self.num_filters = num_filters

        self.conv = nn.ModuleList([CustomConvLayer(node_features, hidden_dim, pos_dim=pos_dim, edge_dim=edge_dim) for _ in range(num_filters)])

        self.predictor = pyg_nn.MLP([hidden_dim * num_filters, hidden_dim, output_dim], bias=[False, True])

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        
        # ConvLayer layers
        x_list = [conv(edge_index=edge_index, x=x, pos=pos) for conv in self.conv]
        x = torch.cat(x_list, dim=-1)
        
        # Global Pooling:
        x = pyg_nn.global_add_pool(x, batch)
        
        # Predictor
        return self.predictor(x)

class CustomGNN_3(nn.Module):
    def __init__(self, node_features=1, pos_dim=3, edge_dim=0, hidden_dim=128, output_dim=1, interaction_nums=5):
        super().__init__()
        self.interaction_nums = interaction_nums
        self.conv1 = CustomConvLayer(node_features, hidden_dim, pos_dim=pos_dim, edge_dim=edge_dim)
        self.conv = nn.ModuleList([CustomConvLayer(hidden_dim, hidden_dim, pos_dim=1, edge_dim=edge_dim) for _ in range(interaction_nums)])

        self.predictor = pyg_nn.MLP([hidden_dim, hidden_dim, output_dim], bias=[False, True])

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        
        out = []
        # First ConvLayer layer
        x = self.conv1(edge_index=edge_index, x=x, pos=pos, 
                    #    edge_attr=edge_attr
                       )
        out.append(x.sigmoid())

        for i in range(self.interaction_nums):
            x = self.conv[i](edge_index, x=x, pos=pos, 
                        #    edge_attr=edge_attr
                        )
            out.append(x.sigmoid())

        out_tensor = torch.stack(out, dim=0)
        x = torch.mean(out_tensor, dim=0)
        # Global Pooling:
        x = pyg_nn.global_add_pool(x, batch)
        
        # Predictor
        return self.predictor(x)

class GATNet(nn.Module):
    def __init__(self, node_features=4, hidden_dim=64, output_dim=1, heads=4, dropout_rate=0.1):
        super(GATNet, self).__init__()

        # GATConv layers
        self.conv1 = pyg_nn.GATConv(node_features, hidden_dim, heads=heads, concat=True, dropout=dropout_rate, aggr="add")
        self.conv2 = pyg_nn.GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, aggr="add")
        
        # predictor
        self.predictor = pyg_nn.MLP([hidden_dim, hidden_dim, output_dim], bias=[False, True])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # First GATConv layer
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        # Second GATConv layer
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x) 

        # Global Pooling
        x = pyg_nn.global_add_pool(x, batch)

        # Predictor
        x = self.predictor(x)

        return x

class DimeNetModel(torch.nn.Module):
    def __init__(self):
        super(DimeNetModel, self).__init__()
        self.model = pyg_nn.DimeNetPlusPlus(hidden_channels=64, out_channels=1, num_blocks=4, int_emb_size=32, basis_emb_size=8, out_emb_channels=128, num_spherical=4, num_radial=4, cutoff=4.0)

    def forward(self, data):
        return self.model(data.z, data.pos, data.batch)

## Data transforms
class ConcatPosToX(T.BaseTransform):
    def __call__(self, data):
        data.x = torch.cat([data.x, data.pos], dim=-1)
        return data

class KNNGroupWithPos(T.BaseTransform):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, data):
        if 'pos' in data:
            edge_index = pyg_nn.knn_graph(data.pos, k=self.k)
        else:
            edge_index = pyg_nn.knn_graph(data.x, k=self.k)
        data.edge_index = edge_index
        return data

class CoulombGroupTransform(T.BaseTransform):
    def __init__(self, k=10):
        self.k = k

    def __call__(self, data):
        # Calculate pairwise Coulomb forces
        num_nodes = data.pos.size(0)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        row, col = edge_index

        pos_diff = data.pos[row] - data.pos[col]
        distances = pos_diff.norm(dim=1).clamp(min=1e-6)
        coulomb_force = (data.z[row] * data.z[col]) / distances

        # Convert dense matrix and find k-nearest neighbors
        force_matrix = torch.zeros((num_nodes, num_nodes))
        force_matrix[row, col] = coulomb_force
        force_matrix[col, row] = coulomb_force

        # Find the k-largest forces
        _, indices = force_matrix.topk(min(num_nodes, self.k), dim=1)

        # Create edge_index
        edge_index = torch.cat([torch.arange(num_nodes)
                                .repeat_interleave(min(num_nodes, self.k))
                                .view(1, -1), indices.view(1, -1)], dim=0)

        data.edge_index = edge_index

        # Add Coulomb force to edge_attr
        row, col = edge_index
        pos_diff = data.pos[row] - data.pos[col]
        distances = pos_diff.norm(dim=1).clamp(min=1e-6)
        coulomb_force = (data.z[row] * data.z[col]) / distances
        data.edge_attr = coulomb_force.view(-1, 1)

        return data

class PruneZeroCharge(T.BaseTransform):
    def __call__(self, data):
        # Keep only non-zero charge nodes
        mask = data.z != 0
        data.x = data.x[mask]
        data.pos = data.pos[mask]
        data.z = data.z[mask]
        return data

class QM7(pyg_data.InMemoryDataset):
    def __init__(self, root, fold=0, train=True,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=True):
        self.train = train
        self.fold = fold
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + '/raw/qm7.mat']
    @property
    def processed_file_names(self):
        if self.train:
            return [f"fold_{self.fold}_train.pt"]
        else:
            return [f"fold_{self.fold}_test.pt"]

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            dataset = scipy.io.loadmat(raw_path)
            data_train, data_val = get_cv_fold(dataset, self.fold)
            if self.fold != None:
                if self.train:
                    dataset = data_train
                else:
                    dataset = data_val

            # print("Keys in the .mat file:", dataset.keys())
            # print("Confirming shape of each field:")
            # print("Coulomb matrices X:", dataset["X"].shape)
            # print("Atomization Energies T:", dataset["T"].shape)
            # print("Atomic Charge of Each Atom Z:", dataset["Z"].shape)
            # print("Cartesian Coordinate of Each Atom R:", dataset["R"].shape)

            # Access the arrays stored in the NPZ file
            z = dataset["Z"]
            z = torch.tensor(z, dtype=torch.int64)
            
            # One-hot encode
            unique_z = torch.unique(z)
            num_classes = unique_z.size(0)
            value_to_index = {v.item(): i for i, v in enumerate(unique_z)}
            z_mapped = z.clone()
            for orig_value, new_index in value_to_index.items():
                z_mapped[z == orig_value] = new_index
            z_one_hot = F.one_hot(z_mapped, num_classes=num_classes).float()
            
            # num_classes = torch.max(z).item() + 1
            # z_one_hot = F.one_hot(z, num_classes=num_classes)
            
            # z_one_hot = z_one_hot.float()
            # x = z_one_hot
            
            x = torch.concat([F.normalize(z.float()).unsqueeze(-1), z_one_hot], dim=-1)
            
            pos = torch.tensor(dataset["R"], dtype=torch.float)
            
            y = torch.tensor(dataset["T"]).unsqueeze(-1)
            
            for index, _ in enumerate(pos):
                data = pyg_data.Data(
                    x = x[index], # for CustomGNN, GATNet
                    pos = pos[index], # for DimeNet, CustomGNN
                    z = z[index], # for DimeNet / torch.Size([23]) (different from others node attr)
                    y = y[index],
                    # Other
                    z_one_hot = z_one_hot[index],
                    )
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

        return data_list
    
## PT Lightning Model
class RMSE(pl.LightningModule):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_pred, y_true):
        mse = F.mse_loss(y_pred, y_true, reduction='mean')
        rmse = torch.sqrt(mse)
        return rmse
    
class GNNPL(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01, batch_size=64, scheduler=None):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.criterion = torch.nn.L1Loss()
        
        self.rmse = RMSE()
        
    def forward(self, data):
        return self.model(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        linear_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max((1 - epoch / 50), 0.02))
        
        exp_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max((0.1 ** (epoch/30)), 0.02))

        plataeau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2)
        lr_scheduler = None
        if self.scheduler != None:
            lr_scheduler = {
                'monitor': 'train_mae', 
                'interval': 'epoch',
                'frequency': 1
            }
            if self.scheduler == 'exp':
                lr_scheduler['scheduler'] = exp_scheduler
            elif self.scheduler == 'linear':
                lr_scheduler['scheduler'] = linear_scheduler
            elif self.scheduler == 'plataeau':
                lr_scheduler['scheduler'] = plataeau_scheduler
            else:
                raise ValueError(f'Unknown scheduler {self.scheduler}')

        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', float(f"{lr:.5e}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('train_mae', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        
        rmse = self.rmse(logits.squeeze(), data.y)
        self.log('train_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        return loss

    def validation_step(self, data, batch_idx):
        
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('val_mae', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        rmse = self.rmse(logits.squeeze(), data.y)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
    def test_step(self, data, batch_idx):
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('test_mae', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        rmse = self.rmse(logits.squeeze(), data.y)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

class CustomProgressBar(pl.callbacks.ProgressBarBase):
    def get_metrics(self, trainer, model):
        # Remove v_num from the progress bar metrics
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
