import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self,output_dim, num_heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(2048,1024, heads=num_heads)
        self.gat2 = GATConv(1024 * num_heads,64, heads=1, concat=False)
        # self.dropout = dropout
        # self.conv2 = GCNConv(1024,512)
        # self.conv3 = GCNConv(512,128)
        # self.conv4 = GCNConv(128, 16)
        self.fc1 = nn.Linear(420*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, data): 
        
        x = data.x
        # print(x)
        edge_index = data.edge_index
        # torch.save(x, 'input_2.txt')
        # for i in batch:
        #     print(i, end=' ')
        # print()
        # x = torch.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.gat1(x, edge_index))

        # torch.save(x, 'GAT_1_2.txt')
        # x = self.dropout(x)
        # x = torch.dropout(x, p=self.dropout, training=self.training)
        # x = torch.relu(self.gat2(x, edge_index))

        x, y = self.gat2(x, edge_index, return_attention_weights=True)

        x = torch.relu(x)
        # torch.save(x, 'GAT_2_2.txt')
        # print(y)
        # with open('./attentionweightsGAT.txt', 'w+') as f:
        #     for i in range(y[0].shape[1]):
        #         f.write(f'{y[0][0][i]} {y[0][1][i]} {y[1][i][0]:.4f}\n')
        # x = self.dropout(x)
        x = x.reshape((-1,420*64))
        # torch.save(x, 'flatten_2.txt')
        # x = x.reshape((-1,420, 512))
        # x = x.sum(dim = 1)
        # print(x)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)

        # torch.save(x, 'FC_2.txt')
        # print(x)
        return x

