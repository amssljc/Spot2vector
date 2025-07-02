# Spot2Vector: extract spot embeddings for spatial transcriptomes data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


class LinearCoder(nn.Module):
    def __init__(self, layer_dims=None):
        super().__init__()
        if layer_dims is None:
            layer_dims = [32, 256, 512, 5000]
        assert len(layer_dims) >= 2, "#layers must >=2, including input layer and output layer."
        if isinstance(layer_dims, int):
            layer_dims = [layer_dims]
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x))
        embeddings = self.layers[-1](x)
        return embeddings


class GraphAttentionEncoder(nn.Module):
    def __init__(self, layer_dims=None, dropout=0.0, heads=1):
        super().__init__()
        if layer_dims is None:
            layer_dims = [2000, 32]
        assert len(layer_dims) >= 2, "#layers must >=2, including input layer and output layer."
        self.layers = nn.ModuleList()
        self.layer_dims = layer_dims
        self.heads = heads
        self.dropout = dropout
        # Create GAT layers based on the dimensions provided in the list
        for i in range(len(layer_dims) - 1):
            if i != (len(layer_dims) - 2):
                self.layers.append(
                    GCNConv(layer_dims[i], layer_dims[i + 1], add_self_loops=False, bias=False)
                )
            else:
                self.layers.append(
                    GATv2Conv(layer_dims[i], layer_dims[i + 1], heads=heads, concat=False, dropout=dropout,
                              add_self_loops=False, bias=False)
                )

    def forward(self, x, edge_index):
        # Process input through each GAT layer using elu activation function
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class GraphAttentionDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.layers = nn.ModuleList()
        self.encoder = encoder
        # Create decoder layers by reversing the dimensions of the encoder layers
        layer_dims = encoder.layer_dims[::-1]
        for i in range(len(layer_dims) - 1):
            if i == 0:
                self.layers.append(
                    GATv2Conv(layer_dims[i], layer_dims[i + 1], heads=encoder.heads, concat=False,
                              dropout=encoder.dropout, add_self_loops=False, bias=False)
                )
            else:
                self.layers.append(
                    GCNConv(layer_dims[i], layer_dims[i + 1], add_self_loops=False, bias=False)
                )

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x
