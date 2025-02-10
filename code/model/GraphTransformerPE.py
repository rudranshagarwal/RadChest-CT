import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv

class GraphTransformerPE(nn.Module):
    def __init__(self,output_dim, heads):
        super(GraphTransformerPE, self).__init__()
        self.conv1 = TransformerConv(2048, 1024, heads=heads)
        self.conv2 = TransformerConv(1024 * heads, 64, heads=1)
        self.nodePE = nn.Embedding(420,2048)
        self.lobePE = nn.Embedding(5, 2048)
        self.lungPE = nn.Embedding(2, 2048)
        # self.conv2 = GCNConv(1024,512)
        # self.conv3 = GCNConv(512,128)
        # self.conv4 = GCNConv(128, 16)
        # self.laplacianTransform = nn.Linear(420,2,2048)
        self.fc1 = nn.Linear(420*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, data, connectivity): 
        x = data.x
        edge_index = data.edge_index
        
        # print(laplacian.shape)
        # laplacian = self.laplacianTransform(laplacian)
        # laplacian = torch.sum(laplacian, dim=2)
        # print(laplacian.shape)
        # x = x + laplacian.reshape((-1,2048))
        x = x + self.nodePE.weight.repeat(len(x)//420,1)

        for i in range(1, 6):
            x[connectivity.reshape(-1) == i]  += self.lobePE.weight[i-1]
            if i == 1 or i == 2:
                x[connectivity.reshape(-1) == i] += self.lungPE.weight[0]
            else:
                x[connectivity.reshape(-1) == i] += self.lungPE.weight[1]
        x = torch.relu(self.conv1(x, edge_index))
        x, y = self.conv2(x,edge_index, return_attention_weights=True)
        x = torch.relu(x)
        # with open('./attentionweightsGT35936.txt', 'a+') as f:
        #     for i in range(y[0].shape[1]):
        #         f.write(f'{y[0][0][i]} {y[0][1][i]} {y[1][i][0]:.4f}\n')
        # print(y)
        # print(x)
        # x = self.conv2(x, edge_index).relu()
        # x = self.conv3(x, edge_index).relu()
        # x = self.conv4(x, edge_index).relu()
        # print(x.shape)
        x = x.reshape((-1,420*64))
        # print(x)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        # print(x)
        return x