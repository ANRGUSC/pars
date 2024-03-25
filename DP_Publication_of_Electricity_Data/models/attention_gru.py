import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import time
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

import wandb

import warnings

# Ignore warnings
warnings.filterwarnings('ignore')



# Assuming your previous model definition
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

    #def forward(self, values, keys, query):
    #    attn_values = torch.bmm(queries, keys.transpose(1, 2))
     #   attention = torch.nn.functional.softmax(attn_values, dim=2)
        
    ##    out = torch.bmm(attention, values)
    #    return out


    def forward(self, values, keys, query):
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        attn_values = torch.bmm(queries, keys.transpose(1, 2))
        attention = torch.nn.functional.softmax(attn_values, dim=2)
        
        out = torch.bmm(attention, values)
        return out



class AttentionGRU(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_dim, output_dim):
        super(AttentionGRU, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, embed_size)
        self.attention = SelfAttention(embed_size)
        self.gru = nn.GRU(embed_size, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.input_dim)
        x = self.embedding(x)
        x = self.attention(x, x, x)
        _, h_n = self.gru(x)
        out = self.fc_out(h_n.squeeze(0))
        return out







