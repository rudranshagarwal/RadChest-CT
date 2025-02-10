import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformerHierarchy(nn.Module):
    def __init__(self,output_dim, heads):
        super(GraphTransformerHierarchy, self).__init__()
        self.conv1 = TransformerConv(2048, 1024, heads=heads)
        self.conv2 = TransformerConv(1024 * heads, 64, heads=1)

        self.verticalconv1 = TransformerConv(64, 64, heads=1)
        self.lobeconv1 = TransformerConv(64, 64, heads=1)

        self.verticalconv2 = TransformerConv(64, 64, heads=1)
        
        self.lungconv1 = TransformerConv(64, 64, heads=1)       
        self.verticalconv3 = TransformerConv(64, 64, heads=1)
        self.backwardconv = TransformerConv(64, 64, heads=1)

        # self.nodePE = nn.Embedding(428,2048)
        # self.lobePE = nn.Embedding(5, 2048)
        # self.lungPE = nn.Embedding(2, 2048)
        # self.conv2 = GCNConv(1024,512)
        # self.conv3 = GCNConv(512,128)
        # self.conv4 = GCNConv(128, 16)
        self.fc1 = nn.Linear(428*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    

    def forward(self, data,lobedata, lobehorizontaldata,lungedges,lunghorizontaledges, topdata, backwarddata, connectivity): 
        x = data.x
        edge_index = data.edge_index

        edge_indexlobe = lobedata.edge_index
        edge_indexlobehorizontal = lobehorizontaldata.edge_index
        edge_indexlungedges = lungedges.edge_index
        edge_indexlunghorizontaledges = lunghorizontaledges.edge_index
        edge_indextop = topdata.edge_index
        edge_indexbackward = backwarddata.edge_index
        # print(x)
        # x = x + self.nodePE.weight.repeat(len(x)//428,1)
        # for i in range(1, 6):
        #     x[connectivity.reshape(-1) == i]  += self.lobePE.weight[i-1]
        #     if i == 1 or i == 2:
        #         x[connectivity.reshape(-1) == i] += self.lungPE.weight[0]
        #     else:
        #         x[connectivity.reshape(-1) == i] += self.lungPE.weight[1]
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x,edge_index))
        x = torch.relu(self.verticalconv1(x, edge_indexlobe ))
        x = torch.relu(self.lobeconv1(x, edge_indexlobehorizontal))
        x = torch.relu(self.verticalconv2(x, edge_indexlungedges))
        x = torch.relu(self.lungconv1(x, edge_indexlunghorizontaledges))
        x = torch.relu(self.verticalconv3(x, edge_indextop))
        x = torch.relu(self.backwardconv(x, edge_indexbackward))
        # print(x)
        # x = self.conv2(x, edge_index).relu()
        # x = self.conv3(x, edge_index).relu()
        # x = self.conv4(x, edge_index).relu()
        # print(x.shape)
        x = x.reshape((-1,428*64))
        # return x
        # print(x)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        # print(x)
        