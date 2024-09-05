import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self,output_dim, num_heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(2048,1024, heads=num_heads)
        self.gat2 = GATConv(1024 * num_heads,512, heads=1, concat=False)
        # self.dropout = dropout
        # self.conv2 = GCNConv(1024,512)
        # self.conv3 = GCNConv(512,128)
        # self.conv4 = GCNConv(128, 16)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    
    def forward(self, data): 
        x = data.x
        edge_index = data.edge_index
        # x = torch.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.gat1(x, edge_index))
        
        # x = torch.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.gat2(x, edge_index))
        # x = x.reshape((-1,420*64))
        x = x.reshape((-1,420, 512))
        x = x.sum(dim = 1)
        
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

