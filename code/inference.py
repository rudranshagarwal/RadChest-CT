import os
import wandb
import math 

from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_checkpoint
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# from train.GCNtrain import GCNtrain
# from train.MLPtrain import MLPtrain

from model.MLP import MLP
from model.GCN import GCN
from model.GAT import GAT
from model.GraphTransformer import GraphTransformer
from model.GraphTransformerPE import GraphTransformerPE
from model.GraphTransformerEdge import GraphTransformerEdge
from model.GraphTransformerHierarchy import GraphTransformerHierarchy
from model.CTNet import CTNet

from dataset.DatasetGCN import CustomDataset as DatasetGCN 
from dataset.DatasetGCNHierarchy import CustomDataset as DatasetGCNHierarchy
from dataset.DatasetCTNet import DatasetCTNet
from utils import get_config, set_seed, read_file, get_counts
from metrics import evaluate_multilabel_classification

# from train import train
# from test import test
def GCNtest(model, traindataloader, criterion, traindataset, optimizer, device, num_epochs, labels, valdataset, valdataloader, num_workers,  testdataloader, testdataset, trainfreq, valfreq,testfreq):
    def save_model_parameters(model, file_path):
        torch.save(model.state_dict(), file_path)
        print(f"Model parameters saved to {file_path}")
    mean = torch.load('../mean6.pth')
    std = torch.load('../std6.pth')
    # Specify the output file path for saving parameters
    # output_file = 'model_parameters_2.pth'

    # Save the model parameters
    # save_model_parameters(model, output_file)
    # model.eval()  
    # with torch.no_grad():
    #     best_val_map = -1
    #     best_val_map_epoch = -1
    #     running_loss = 0.0
    #     running_loss2 = 0.0
    #     train_loss = 0
    #     val_loss = 0
    #     epoch_outputs = torch.empty((0,len(labels)))
    #     # epoch_outputs2 = torch.empty((0,420, 2048))
    #     epoch_targets = torch.empty((0,len(labels)))
    #     for inputs, targets, edges, image in traindataloader:
    #         mean = inputs.mean(dim=2, keepdim=True)[0]
    #         std = inputs.std(dim=2, keepdim=True)[0]
    #         inputs = (inputs - mean) / (std + 1e-8) 
    #         # print(inputs.dtype)
    #         graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
    #         dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
    #         batchx = []
    #         for databatch in dataloader_graph:
    #             batchx = databatch
    #         # print(inputs.shape)
    #         # inputs = inputs.to(device)
    #         batchx = batchx.to(device)
    #         targets = targets.to(device)
    #         # edges = edges.to(device)
    #         # optimizer.zero_grad()
    #         # outputs = model(inputs, edges)
    #         outputs = model(batchx)
    #         # outputs2 = model(batchx)
    #         # print(outputs)
    #         # print(targets)
    #         loss = criterion(outputs, targets)
    #         # loss2 = criterion(outputs2, targets)
    #         # print(loss.item())
    #         # loss.backward()
    #         # loss2.backward()
    #         # optimizer.step()
    #         epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
    #         # epoch_outputs2 = torch.cat((epoch_outputs2, inputs.cpu()))
    #         epoch_targets = torch.cat((epoch_targets, targets.cpu()))
    #         running_loss += loss.item() * inputs.size(0)
    #         # running_loss2 += loss2.item() * inputs.size(0)
    #         # break
    #     # print(running_loss)
    #     epoch_loss = running_loss / len(traindataset)
    #     # print(running_loss2/len(traindataset))
    #     train_loss = epoch_loss
    #         # break
    #     # print(epoch_outputs)
    #     probabilities = torch.sigmoid(epoch_outputs)
    #     # torch.save(epoch_outputs, './test1.pth')
    #     # torch.save(epoch_outputs2, './test2.pth')
    #     # torch.save(epoch_targets, 'targets2.pth')
    #     predictions = (probabilities > 0.5).float()

    #     predictions_np = predictions.detach().numpy()
    #     targets_np = epoch_targets.numpy()

    #     print(f"Train metrics :\n")
    #     train_accuracy, train_map, train_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(), labels, trainfreq, len(traindataset))
    #     print(f'Train Loss: {train_loss:.4f}\n')

    # model.eval()

    # with torch.no_grad():
    #     running_loss = 0.0
    #     epoch_outputs = torch.empty((0,len(labels)))
    #     epoch_targets = torch.empty((0,len(labels)))
    #     for inputs, targets, edges, image in valdataloader:
    #         mean = inputs.mean(dim=2, keepdim=True)[0]
    #         std = inputs.std(dim=2, keepdim=True)[0]
    #         inputs = (inputs - mean) / (std + 1e-8) 
    #         # print(inputs)
    #         graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
    #         dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
    #         batchx = []
    #         for databatch in dataloader_graph:
    #             batchx = databatch
    #         # print(inputs.shape)
    #         # inputs = inputs.to(device)
    #         batchx = batchx.to(device)
    #         targets = targets.to(device)
    #         # edges = edges.to(device)
    #         # outputs = model(inputs, edges)
    #         outputs = model(batchx)
    #         # print(outputs)
    #         loss = criterion(outputs, targets)
    #         epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
    #         epoch_targets = torch.cat((epoch_targets, targets.cpu()))
    #         running_loss += loss.item() * inputs.size(0)
    #     epoch_loss = running_loss / len(valdataset)
    #     val_loss = epoch_loss
    #     probabilities = torch.sigmoid(epoch_outputs)
    #     predictions = (probabilities > 0.5).float()

    #     predictions_np = predictions.detach().numpy()
    #     targets_np = epoch_targets.numpy()
    #     print(f"Val metrics:\n")
    #     val_accuracy, val_map, val_auc = evaluate_multilabel_classification(targets_np, predictions_np , probabilities.detach().numpy(),labels, valfreq, len(valdataset))
    #     print(f'Val Loss: {val_loss:.4f}\n')
    

    model.eval()

    
    running_loss = 0.0
    test_loss = 0
    epoch_outputs = torch.empty((0,len(labels)))
    epoch_targets = torch.empty((0,len(labels)))
    for inputs, targets, edges, image, connectivity in testdataloader:
        # print(image)
        # mean = inputs.mean(dim=2, keepdim=True)
        # std = inputs.std(dim=2, keepdim=True)
        inputs = (inputs - mean) / (std + 1e-8) 
        # print(inputs)
        graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
        dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        batchx = []
        for databatch in dataloader_graph:
            # print(databatch.x.is_leaf)
            batchx = databatch
            # print(batchx.x.is_leaf)
        # print(inputs.shape)
        # inputs = inputs.to(device)
        # print(batchx.x[6720])
        # batchx.requires_grad = True
        # batchx.x.requires_grad = True
        # print(batchx.x.is_leaf)

        batchx = batchx.to(device)
        # batchx.x.retain_grad()
        targets = targets.to(device)
        # edges = edges.to(device)
        # outputs = model(inputs, edges)
        outputs = model(batchx)
        # target_class = 2
        # target_score = outputs[0,target_class]
        # print(target_score)
        
        # target_score.backward()
        # print(batchx.x)
        # saliency = batchx.x.grad.data.abs()
        # torch.save(saliency, 'saliencyGTtst35936.pth')
        # print(outputs)
        loss = criterion(outputs, targets)
        epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
        epoch_targets = torch.cat((epoch_targets, targets.cpu()))
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(testdataset)
    test_loss = epoch_loss
    probabilities = torch.sigmoid(epoch_outputs)
    predictions = (probabilities > 0.5).float()
    # print(probabilities[:, 2])
    # for i in range(len(probabilities)):
    #     print(probabilities[i][-2])
    predictions_np = predictions.detach().numpy()
    targets_np = epoch_targets.numpy()
    print("Test metrics:\n")
    test_accuracy, test_map, test_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(),labels, testfreq, len(testdataset))
    print(f'Test Loss: {test_loss:.4f}\n')
    
        
def GCNtesthierarchy(model, traindataloader, criterion, traindataset, optimizer, device, num_epochs, labels, valdataset, valdataloader, num_workers,  testdataloader, testdataset, trainfreq, valfreq,testfreq):
    def save_model_parameters(model, file_path):
        torch.save(model.state_dict(), file_path)
        print(f"Model parameters saved to {file_path}")
    mean = torch.load('../mean6.pth')
    std = torch.load('../std6.pth')
   

    model.eval()

    
    running_loss = 0.0
    test_loss = 0
    epoch_outputs = torch.empty((0,len(labels)))
    epoch_targets = torch.empty((0,len(labels)))
    for inputs, targets, edges, edges2, edges3, edges4, edges5,edges6, edges7,image, connectivity in testdataloader:

        inputs = (inputs - mean) / (std + 1e-8) 
    
        graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
        graphs2 = [Data(x = inputs[j], edge_index = edges2[j]) for j in range(inputs.shape[0])]
        graphs3 = [Data(x = inputs[j], edge_index = edges3[j]) for j in range(inputs.shape[0])]
        graphs4 = [Data(x = inputs[j], edge_index = edges4[j]) for j in range(inputs.shape[0])]
        graphs5 = [Data(x = inputs[j], edge_index = edges5[j]) for j in range(inputs.shape[0])]
        graphs6 = [Data(x = inputs[j], edge_index = edges6[j]) for j in range(inputs.shape[0])]
        graphs7 = [Data(x = inputs[j], edge_index = edges7[j]) for j in range(inputs.shape[0])]
        dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        dataloader_graph2 = DataLoader(graphs2, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        dataloader_graph3 = DataLoader(graphs3, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        dataloader_graph4 = DataLoader(graphs4, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        dataloader_graph5 = DataLoader(graphs5, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        dataloader_graph6 = DataLoader(graphs6, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        dataloader_graph7 = DataLoader(graphs7, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
        batchx = []
        for databatch in dataloader_graph:
            batchx = databatch
        batchx2 = []
        for databatch in dataloader_graph2:
            batchx2 = databatch
        batchx3 = []
        for databatch in dataloader_graph3:
            batchx3 = databatch
        batchx4 = []
        for databatch in dataloader_graph4:
            batchx4 = databatch
        batchx5 = []
        for databatch in dataloader_graph5:
            batchx5 = databatch
        batchx6 = []
        for databatch in dataloader_graph6:
            batchx6 = databatch
        batchx7 = []
        for databatch in dataloader_graph7:
            batchx7 = databatch
        
        batchx = batchx.to(device)
        batchx2 = batchx2.to(device)
        batchx3 = batchx3.to(device)
        batchx4 = batchx4.to(device)
        batchx5 = batchx5.to(device)
        batchx6 = batchx6.to(device)
        batchx7 = batchx7.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(batchx, batchx2, batchx3, batchx4, batchx5,batchx6,batchx7, connectivity)
        # target_class = 2
        # target_score = outputs[0,target_class]
        # print(target_score)
        
        # target_score.backward()
        # print(batchx.x)
        # saliency = batchx.x.grad.data.abs()
        # torch.save(saliency, 'saliencyGTtst35936.pth')
        # print(outputs)
        loss = criterion(outputs, targets)
        epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
        epoch_targets = torch.cat((epoch_targets, targets.cpu()))
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(testdataset)
    test_loss = epoch_loss
    probabilities = torch.sigmoid(epoch_outputs)
    torch.save(probabilities, 'GCNhierarchyprobabilitieshierarchy.pth')
    torch.save(epoch_targets, 'GCNhierarchygroundtruth.pth')
    predictions = (probabilities > 0.5).float()
    # print(probabilities[:, 2])
    # for i in range(len(probabilities)):
    #     print(probabilities[i][-2])
    predictions_np = predictions.detach().numpy()
    targets_np = epoch_targets.numpy()
    print("Test metrics:\n")
    test_accuracy, test_map, test_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(),labels, testfreq, len(testdataset))
    print(f'Test Loss: {test_loss:.4f}\n')


def MLPtest(model, traindataloader, criterion, traindataset, optimizer, device, num_epochs, labels, valdataset, valdataloader, num_workers, testdataloader, testdataset, trainfreq, valfreq,testfreq):
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # model.eval()  
    # best_val_map = -1
    # best_val_map_epoch = -1
    # with torch.no_grad():
    #     running_loss = 0.0
    #     train_loss = 0
    #     val_loss = 0
    #     epoch_outputs = torch.empty((0,len(labels)))
    #     epoch_targets = torch.empty((0,len(labels)))
    #     for inputs, targets, image in traindataloader:
    #         # mean = inputs.mean(dim=2, keepdim=True)[0]
    #         # std = inputs.std(dim=2, keepdim=True)[0]
    #         # inputs = (inputs - mean) / (std + 1e-8) 
    #         # print(inputs)
    #         # print(inputs.shape)
    #         # inputs = inputs.to(device)
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         # edges = edges.to(device)
    #         optimizer.zero_grad()
    #         # outputs = model(inputs, edges)
    #         outputs = model(inputs)
    #         # print(outputs)
    #         loss = criterion(outputs, targets)
    #         epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
    #         epoch_targets = torch.cat((epoch_targets, targets.cpu()))
    #         running_loss += loss.item() * inputs.size(0)
    #         epoch_loss = running_loss / len(traindataset)
    #         train_loss = epoch_loss
    #     probabilities = torch.sigmoid(epoch_outputs)
    #     predictions = (probabilities > 0.5).float()

    #     predictions_np = predictions.detach().numpy()
    #     targets_np = epoch_targets.numpy()
    #     print(probabilities)
    #     print(f"Train metrics for epoch:\n")
    #     train_accuracy, train_map, train_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(), labels, trainfreq, len(traindataset))
    #     print(f'Train Loss: {train_loss:.4f}\n')
    # model.eval()

    # with torch.no_grad():
    #     running_loss = 0.0
    #     epoch_outputs = torch.empty((0,len(labels)))
    #     epoch_targets = torch.empty((0,len(labels)))
    #     for inputs, targets, image in valdataloader:
    #         # mean = inputs.mean(dim=2, keepdim=True)[0]
    #         # std = inputs.std(dim=2, keepdim=True)[0]
    #         # inputs = (inputs - mean) / (std + 1e-8) 
    #         # print(inputs)
    #         # print(inputs.shape)
    #         inputs = inputs.to(device)
    #         # batchx = batchx.to(device)
    #         targets = targets.to(device)
    #         # edges = edges.to(device)
    #         # outputs = model(inputs, edges)
    #         outputs = model(inputs)
    #         # print(outputs)
    #         loss = criterion(outputs, targets)
    #         epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
    #         epoch_targets = torch.cat((epoch_targets, targets.cpu()))
    #         running_loss += loss.item() * inputs.size(0)
    #         epoch_loss = running_loss / len(valdataset)
    #         val_loss = epoch_loss
    #     probabilities = torch.sigmoid(epoch_outputs)
    #     predictions = (probabilities > 0.5).float()

    #     predictions_np = predictions.detach().numpy()
    #     targets_np = epoch_targets.numpy()
    #     print(f"Val metrics for epoch:\n")
    #     val_accuracy, val_map, val_auc = evaluate_multilabel_classification(targets_np, predictions_np , probabilities.detach().numpy(),labels, valfreq, len(valdataset))
    #     print(f'Val Loss: {val_loss:.4f}\n')


    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        test_loss = 0
        epoch_outputs = torch.empty((0,len(labels)))
        epoch_targets = torch.empty((0,len(labels)))
        for inputs, targets,  image in testdataloader:
            # mean = inputs.mean(dim=2, keepdim=True)[0]
            # std = inputs.std(dim=2, keepdim=True)[0]
            # inputs = (inputs - mean) / (std + 1e-8) 
            # print(inputs)
            
            # print(inputs.shape)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # edges = edges.to(device)
            # outputs = model(inputs, edges)
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)
            epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
            epoch_targets = torch.cat((epoch_targets, targets.cpu()))
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(testdataset)
            test_loss = epoch_loss
        probabilities = torch.sigmoid(epoch_outputs)
        predictions = (probabilities > 0.5).float()
        print(probabilities)
        predictions_np = predictions.detach().numpy()
        targets_np = epoch_targets.numpy()
        torch.save(probabilities, 'probabilitieshierarchy.pth')
        torch.save(epoch_targets, 'groundtruth.pth')
        print("Test metrics:\n")
        test_accuracy, test_map, test_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(),labels, testfreq, len(testdataset))
        print(f'Test Loss: {test_loss:.4f}\n')


def inference():
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_file(config["labels"])
    testdataset =[]
    testdataloader = []
    criterion = []
    train_frequencies = get_counts(config["labels"], config["dataset"]["train_labels"])
    val_frequencies = get_counts(config["labels"], config["dataset"]["val_labels"])
    test_frequencies = get_counts(config["labels"], config["dataset"]["test_labels"])
    if config["model"] == 'GraphTransformerHierarchy':
        train_dataset = DatasetGCNHierarchy(annotations_file=config["dataset"]["train_labels"], feature_dir=config["dataset"]["train_features"], slic_dir=config["dataset"]["train_slic"], lungmask_dir=config["dataset"]["train_lungmask"], centroids_dir=config["dataset"]["train_centroids"], k=config["models"][config['model']]["k"] ,labels = labels, exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["train_connectivity"], connectivity=config["connectivity"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        val_dataset = DatasetGCNHierarchy(annotations_file=config["dataset"]["val_labels"], feature_dir=config["dataset"]["val_features"], slic_dir=config["dataset"]["val_slic"], lungmask_dir=config["dataset"]["val_lungmask"], centroids_dir=config["dataset"]["val_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["val_connectivity"], connectivity=config["connectivity"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        test_dataset = DatasetGCNHierarchy(annotations_file=config["dataset"]["test_labels"], feature_dir=config["dataset"]["test_features"], slic_dir=config["dataset"]["test_slic"], lungmask_dir=config["dataset"]["test_lungmask"], centroids_dir=config["dataset"]["test_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["test_connectivity"], connectivity=config["connectivity"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
       
        
        model = GraphTransformerHierarchy(len(labels), config["models"]["GraphTransformerHierarchy"]["num_heads"]).to(device)
        weights = [1/train_frequencies[i] for i in train_frequencies.keys()]
        weights = torch.tensor(weights)
        # print(weights)
        # weights /= weights.sum()

        # weights = torch.tensor([(1 - config["beta"])/(1 - config["beta"] ** train_frequencies[i]) for i in train_frequencies.keys()])
        # weights = torch.tensor([(math.log(1 + len(train_dataset)/train_frequencies[i])) for i in train_frequencies.keys()])
        # print(weights)
        weights *= len(train_dataset)
        # print(train_frequencies)
        print(weights)
        # criterion = nn.BCEWithLogitsLoss(pos_weight= weights.to(device))  
        # criterion = nn.BCEWithLogitsLoss()  
        # optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])
        criterion = nn.BCEWithLogitsLoss(pos_weight= weights.to(device))  
        # criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"/scratch/RadChest/checkpoints/AblationNoLungConvmodel_GraphTransformerHierarchy_lr_2e-05_bs_64_epochs_100_labels_17connectivity=allk=5_10-02-2025_17-04-46/model_18.pth", device)
        # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.99, weight_decay=1e-7)
        GCNtesthierarchy(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'],testdataloader=test_loader, testdataset=test_dataset, trainfreq = train_frequencies, valfreq=val_frequencies, testfreq = test_frequencies)
    elif config["model"][0] == 'G':
        train_dataset = DatasetGCN(annotations_file=config["dataset"]["train_labels"], feature_dir=config["dataset"]["train_features"], slic_dir=config["dataset"]["train_slic"], lungmask_dir=config["dataset"]["train_lungmask"], centroids_dir=config["dataset"]["train_centroids"], k=config["models"][config['model']]["k"] ,labels = labels, exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["train_connectivity"], connectivity=config["connectivity"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

        val_dataset = DatasetGCN(annotations_file=config["dataset"]["val_labels"], feature_dir=config["dataset"]["val_features"], slic_dir=config["dataset"]["val_slic"], lungmask_dir=config["dataset"]["val_lungmask"], centroids_dir=config["dataset"]["val_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]),connectivity_dir=config["dataset"]["val_connectivity"], connectivity=config["connectivity"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        test_dataset = DatasetGCN(annotations_file=config["dataset"]["test_labels"], feature_dir=config["dataset"]["test_features"], slic_dir=config["dataset"]["test_slic"], lungmask_dir=config["dataset"]["test_lungmask"], centroids_dir=config["dataset"]["test_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["test_connectivity"], connectivity=config["connectivity"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        if(config["model"] == "GCN"):
            model = GCN(len(labels)).to(device)
        elif(config["model"] == "GAT"):
            model = GAT(len(labels), config["models"]["GAT"]["num_heads"]).to(device)
        elif(config["model"] == "GraphTransformer"):
           model = GraphTransformer(len(labels), config["models"]["GraphTransformer"]["num_heads"]).to(device)
        elif(config["model"] == "GraphTransformerPE"):
           model = GraphTransformerPE(len(labels), config["models"]["GraphTransformerPE"]["num_heads"]).to(device)
        elif(config["model"] == "GraphTransformerEdge"):
            model = GraphTransformerEdge(len(labels), config["models"]["GraphTransformerEdge"]["num_heads"],  config["models"]["GraphTransformerEdge"]["edge_dim"]).to(device)
        weights = [1/train_frequencies[i] for i in train_frequencies.keys()]
        weights = torch.tensor(weights)
        # weights /= weights.sum()

        # weights = torch.tensor([(1 - config["beta"])/(1 - config["beta"] ** train_frequencies[i]) for i in train_frequencies.keys()])
        # weights = torch.tensor([(math.log(1 + len(train_dataset)/train_frequencies[i])) for i in train_frequencies.keys()])
        weights *= len(train_dataset)
        # print(train_frequencies)
        print(weights)
        criterion = nn.BCEWithLogitsLoss(pos_weight= weights.to(device))  
        optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"/scratch/RadChest/checkpoints/model_GraphTransformerEdge_lr_2e-05_bs_64_epochs_100_labels_17connectivity=allk=5_31-12-2024_00-55-12/model_7.pth", device)
        # criterion = nn.BCEWithLogitsLoss()  

        # save_dir = os.path.join(config["save_dir"], run_name)
        GCNtest(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'],testdataloader=test_loader, testdataset=test_dataset, trainfreq = train_frequencies, valfreq=val_frequencies, testfreq = test_frequencies)
    
    else:
        if(config["model"] == "CTNet"):
            model = CTNet(len(labels)).to(device)
            optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])
            train_dataset = DatasetCTNet(annotations_file=config["dataset"]["train_labels"], image_dir=config["dataset"]["train_images"], lungmask_dir=config["dataset"]["train_lungmask"],labels = labels, exclude=read_file(config["exclude"]), split="train", data_augment=False)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

            val_dataset = DatasetCTNet(annotations_file=config["dataset"]["val_labels"], image_dir=config["dataset"]["val_images"], lungmask_dir=config["dataset"]["val_lungmask"] ,labels = labels, exclude=read_file(config["exclude"]), split="val", data_augment=False)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

            test_dataset = DatasetCTNet(annotations_file=config["dataset"]["test_labels"], image_dir=config["dataset"]["test_images"], lungmask_dir=config["dataset"]["test_lungmask"] ,labels = labels, exclude=read_file(config["exclude"]), split="test", data_augment=False)
            test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
           

            model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"/scratch/RadChest/checkpoints/model_CTNet_lr_2e-05_bs_2_epochs_100_labels_17_23-09-2024_10-15-08/model_98.pth", device)

            # model = CTNet(len(labels)).to(device)
        weights = [1/train_frequencies[i] for i in train_frequencies.keys()]
        weights = torch.tensor(weights)
        # weights /= weights.sum()

        # weights = torch.tensor([(1 - config["beta"])/(1 - config["beta"] ** train_frequencies[i]) for i in train_frequencies.keys()])
        # weights = torch.tensor([(math.log(1 + len(train_dataset)/train_frequencies[i])) for i in train_frequencies.keys()])
        # weights *= len(train_dataset)
        # print(train_frequencies)
        # print(weights)
        criterion = nn.BCEWithLogitsLoss(pos_weight= weights.to(device))
        # criterion = nn.BCEWithLogitsLoss()  
        MLPtest(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'], testdataloader=test_loader, testdataset=test_dataset, trainfreq = train_frequencies, valfreq=val_frequencies, testfreq = test_frequencies)


    


    
if __name__ == "__main__":
    config = get_config("config.yaml")

    inference()

    # wandb.finish()