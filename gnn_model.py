"""GraphSAGE dirigido de dos capas + clasificación de transferencias."""
import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class AMLGraphSAGE(nn.Module):
    def __init__(self, edge_features, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(4, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden + edge_features, hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index).relu()

    def score(self, z, pairs, attrs):
        return self.head(torch.cat([z[pairs[0]], z[pairs[1]], attrs], dim=1)).flatten()
