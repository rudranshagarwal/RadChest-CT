import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformerEdge(nn.Module):
    def __init__(self,output_dim, heads, edge_dim):
        super(GraphTransformerEdge, self).__init__()
        self.conv1 = TransformerConv(2048, 1024, heads=heads, edge_dim=edge_dim)
        self.conv2 = TransformerConv(1024 * heads, 64, heads=1, edge_dim=edge_dim)
        self.edge_features = nn.Parameter(torch.rand((5*420, edge_dim)))
        # self.lobePE = nn.Embedding(5, 2048)
        # self.conv2 = GCNConv(1024,512)
        # self.conv3 = GCNConv(512,128)
        # self.conv4 = GCNConv(128, 16)
        self.fc1 = nn.Linear(420*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    
    def forward(self, data): 
        x = data.x
        edge_index = data.edge_index
        # print(x)
        # print(self.edge_features.shape)
        # for i in range(1, 6):
        #     x[connectivity.reshape(-1) == i]  += self.lobePE.weight[i-1]
        x = torch.relu(self.conv1(x, edge_index, self.edge_features.repeat(len(x)//(420),1)))
        x = torch.relu(self.conv2(x,edge_index, self.edge_features.repeat(len(x)//(420),1)))
        # print(x)
        # x = self.conv2(x, edge_index).relu()
        # x = self.conv3(x, edge_index).relu()
        # x = self.conv4(x, edge_index).relu()
        # print(x.shape)
        x = x.reshape((-1,420*64))
        return x
        # print(x)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # print(x)
        return x