import torch
from torch.nn import BatchNorm1d, Linear
import torch.nn.functional as F

import pytorch_lightning as L

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import dropout_edge

# from .rev_gat import *

# class GATNet(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
#         super(GATNet, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
#         self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, dropout=0.6):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Create the first GATConv layer
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
            
            # Create the intermediate GATConv layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
            
            # Create the last GATConv layer
            self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # Apply ELU and dropout to all but the last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=1)

class GATNet2(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_heads,
                 num_layers,
                 dropout=0.6,
                 edge_dropout=0.4,
                 input_dropout=0.35,
                 input_norm=False):
        super(GATNet2, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.input_dropout = input_dropout

        # Input normalization
        if input_norm:
            self.input_norm = BatchNorm(in_channels)

        # GAT layers with normalization and dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.residuals = torch.nn.ModuleList()

        for i in range(num_layers):
            in_hidden = hidden_channels * num_heads if i > 0 else in_channels
            out_hidden = hidden_channels if i < num_layers - 1 else out_channels
            num_heads_layer = num_heads if i < num_layers - 1 else 1
            self.convs.append(GATConv(in_hidden, out_hidden, heads=num_heads_layer, concat=True if i < num_layers - 1 else False, dropout=dropout))
            self.bns.append(BatchNorm1d(out_hidden * num_heads_layer))
            if i > 0:
                self.residuals.append(Linear(in_hidden, out_hidden * num_heads_layer if i < num_layers - 1 else out_channels))

        # Output processing
        self.linear = Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        if hasattr(self, "input_norm"):
            x = self.input_norm(x)

        x = F.dropout(x, p=self.input_dropout, training=self.training)

        # Apply edge dropout
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            if x.shape[0] > 1:
                self.bns[i](x)
            x = F.relu(x)
            if i > 0:
                x = x + self.residuals[i-1](residual)  # Add residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

# class GATNet2(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, dropout=0.6, input_norm=False):
#         super(GATNet2, self).__init__()
#         self.num_layers = num_layers
#         self.dropout = dropout

#         # Create the GATConv layers
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()  # BatchNorm layers

#         if input_norm:
#             self.input_norm = BatchNorm1d(in_channels * num_heads)

#         if num_layers == 1:
#             self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
#         else:
#             self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
#             self.bns.append(BatchNorm1d(hidden_channels * num_heads))
            
#             # Create the intermediate GATConv layers
#             for _ in range(num_layers - 2):
#                 self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
#                 self.bns.append(BatchNorm1d(hidden_channels * num_heads))
            
#             # Create the last GATConv layer
#             self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        
#         # Small linear layer at the end
#         self.linear = Linear(out_channels, out_channels)

#     def forward(self, x, edge_index):
#         if hasattr(self, "input_norm"):
#             x = self.input_norm(x)
            
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if i < self.num_layers - 1:  # Apply ELU, BatchNorm, and dropout to all but the last layer
#                 x = F.elu(x)
#                 x = self.bns[i](x)
#                 #x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.linear(x)  # Apply the small linear layer
#         return F.log_softmax(x, dim=1)
        
class LitGAT(L.LightningModule):
    def __init__(self, model, criterion, learning_rate=0.005, weight_decay=5e-4):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        train_mask = batch.train_mask
        loss = self.criterion(out[train_mask], batch.y[train_mask].squeeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_mask = batch.val_mask
        out = self(batch.x, batch.edge_index)
        loss = self.criterion(out[val_mask], batch.y[val_mask].squeeze(1))
        self.log('val_loss', loss)
        # acc = self.compute_accuracy(out, batch.y, val_mask)
        # self.log('val_acc', acc)
        # just to check
        acc = self.compute_accuracy2(out, batch.y, batch.batch_size)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        test_mask = batch.test_mask
        out = self(batch.x, batch.edge_index)
        # acc = self.compute_accuracy(out, batch.y, test_mask)
        # self.log('test_acc', acc)
        # just to check
        acc2 = self.compute_accuracy2(out, batch.y, batch.batch_size)
        self.log('test_acc', acc2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def compute_accuracy_wrong(self, out, y, mask):
        # WRONG FOR BATCH TRAINING
        # This counts double, triple etc
        pred = out.argmax(dim=1)
        correct = pred[mask] == y[mask].squeeze(1)
        acc = int(correct.sum()) / int(mask.sum())
        return acc
    
    # just to check:
    def compute_accuracy2(self, out, y, batch_size):
        pred = out.argmax(dim=1)
        correct = pred[:batch_size] == y[:batch_size].squeeze(1)
        acc = int(correct.sum()) / int(batch_size)
        return acc

# not very elegant; maybe fix later
class LitGAT_only_labels(LitGAT):
    def __init__(self, model, criterion, learning_rate=0.005, weight_decay=5e-4):
        super().__init__(model, criterion, learning_rate, weight_decay)

    def get_out_and_mask(self, batch, og_mask):
        # masking input nodes
        masked_x = batch.x.clone().float()  # Clone the node features which are in this case labels
        input_ids = batch.input_id
        indices_tensor = torch.nonzero((batch.n_id[:, None] == input_ids).any(dim=1))
        masked_x[indices_tensor] = -1  # Zero out the features of the input node to mask the label
        # correcting mask
        mask = torch.zeros_like(og_mask, dtype=torch.bool)
        mask[indices_tensor] = True
        return self(masked_x, batch.edge_index), mask

    def training_step(self, batch, batch_idx):
        out, mask = self.get_out_and_mask(batch, batch.train_mask)
        loss = self.criterion(out[mask], batch.y[mask].squeeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out, mask = self.get_out_and_mask(batch, batch.val_mask)
        loss = self.criterion(out[mask], batch.y[mask].squeeze(1))
        acc = self.compute_accuracy(out, batch.y, batch.batch_size)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        out, mask = self.get_out_and_mask(batch, batch.test_mask)
        acc = self.compute_accuracy(out, batch.y, batch.batch_size)
        self.log('test_acc', acc)

    def compute_accuracy(self, out, y, batch_size):
        pred = out.argmax(dim=1)
        correct = pred[:batch_size] == y[:batch_size].squeeze(1)
        acc = int(correct.sum()) / int(batch_size)
        return acc

# def train_gnn(model, train_mask, optimizer, criterion, data: Data):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = criterion(out[train_mask], data.y[train_mask].squeeze(1))
#     loss.backward()
#     optimizer.step()
#     return loss

# def evaluate_gnn(model, val_mask, data: Data, get_label_distribution=False):
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)  # Use the class with highest probability
#         correct = pred[val_mask] == data.y[val_mask].squeeze(1)
#         acc = int(correct.sum()) / int(val_mask.sum())
#         if not get_label_distribution:
#             return acc
#         else:
#             label_distribution = dict()
#             for p_t in pred[val_mask]:
#                 p = int(p_t)
#                 if p in label_distribution.keys():
#                     label_distribution[p] += 1
#                 else:
#                     label_distribution[p] = 1
#             return acc, label_distribution