import os
import wandb
import math 

from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


from train.GCNtrain import GCNtrain
from train.GCNtrainhierarchy import GCNtrainhierarchy
from train.MLPtrain import MLPtrain

from model.MLP import MLP
from model.GCN import GCN
from model.GAT import GAT
from model.GATGT import GATGT
from model.GraphTransformer import GraphTransformer
from model.GraphTransformerEdge import GraphTransformerEdge
from model.GraphTransformerPE import GraphTransformerPE
from model.GraphTransformerHierarchy import GraphTransformerHierarchy
from model.CTNet import CTNet
from model.CTNetNoConv import CTNetNoConv

from dataset.DatasetGCN import CustomDataset as DatasetGCN 
from dataset.DatasetGCNHierarchy import CustomDataset as DatasetGCNHierarchy

from dataset.DatasetCTNet import DatasetCTNet

from dataset.DatasetMLP import DatasetMLP

from utils import get_config, set_seed, read_file, get_counts

# from train import train
# from test import test





def main(run_name):
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_file(config["labels"])

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
       
        
        model = GraphTransformerHierarchy(len(labels), config["models"]["GraphTransformerPE"]["num_heads"]).to(device)
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
        # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.99, weight_decay=1e-7)
        save_dir = os.path.join(config["save_dir"], run_name)
        GCNtrainhierarchy(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'],save_dir=save_dir, testdataloader=test_loader, testdataset=test_dataset, trainfreq = train_frequencies, valfreq=val_frequencies, testfreq = test_frequencies)
    elif config["model"][0] == 'G':
        train_dataset = DatasetGCN(annotations_file=config["dataset"]["train_labels"], feature_dir=config["dataset"]["train_features"], slic_dir=config["dataset"]["train_slic"], lungmask_dir=config["dataset"]["train_lungmask"], centroids_dir=config["dataset"]["train_centroids"], k=config["models"][config['model']]["k"] ,labels = labels, exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["train_connectivity"], connectivity=config["connectivity"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        val_dataset = DatasetGCN(annotations_file=config["dataset"]["val_labels"], feature_dir=config["dataset"]["val_features"], slic_dir=config["dataset"]["val_slic"], lungmask_dir=config["dataset"]["val_lungmask"], centroids_dir=config["dataset"]["val_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["val_connectivity"], connectivity=config["connectivity"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        test_dataset = DatasetGCN(annotations_file=config["dataset"]["test_labels"], feature_dir=config["dataset"]["test_features"], slic_dir=config["dataset"]["test_slic"], lungmask_dir=config["dataset"]["test_lungmask"], centroids_dir=config["dataset"]["test_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]), connectivity_dir=config["dataset"]["test_connectivity"], connectivity=config["connectivity"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
       
        if(config["model"] == "GCN"):
            model = GCN(len(labels)).to(device)
        elif(config["model"] == "GAT"):
            model = GAT(len(labels), config["models"]["GAT"]["num_heads"]).to(device)
        elif(config["model"] == "GraphTransformer"):
            model = GraphTransformer(len(labels), config["models"]["GraphTransformer"]["num_heads"]).to(device)
        elif(config["model"] == "GATGT"):
            model = GATGT(len(labels), config["models"]["GATGT"]["num_heads"]).to(device)
        elif(config["model"] == "GraphTransformerEdge"):
            model = GraphTransformerEdge(len(labels), config["models"]["GraphTransformerEdge"]["num_heads"],  config["models"]["GraphTransformerEdge"]["edge_dim"]).to(device)
        elif(config["model"] == "GraphTransformerPE"):
            model = GraphTransformerPE(len(labels), config["models"]["GraphTransformerPE"]["num_heads"]).to(device)
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
        # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.99, weight_decay=1e-7)
        save_dir = os.path.join(config["save_dir"], run_name)
        GCNtrain(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'],save_dir=save_dir, testdataloader=test_loader, testdataset=test_dataset, trainfreq = train_frequencies, valfreq=val_frequencies, testfreq = test_frequencies)
    
    else:
        if(config["model"] == "CTNet" or config["model"] == "CTNetNoConv"):
            train_dataset = DatasetCTNet(annotations_file=config["dataset"]["train_labels"], image_dir=config["dataset"]["train_images"], lungmask_dir=config["dataset"]["train_lungmask"],labels = labels, exclude=read_file(config["exclude"]), split="train", data_augment=False)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

            val_dataset = DatasetCTNet(annotations_file=config["dataset"]["val_labels"], image_dir=config["dataset"]["val_images"], lungmask_dir=config["dataset"]["val_lungmask"] ,labels = labels, exclude=read_file(config["exclude"]), split="val", data_augment=False)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

            test_dataset = DatasetCTNet(annotations_file=config["dataset"]["test_labels"], image_dir=config["dataset"]["test_images"], lungmask_dir=config["dataset"]["test_lungmask"] ,labels = labels, exclude=read_file(config["exclude"]), split="test", data_augment=False)
            test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
        
            if config["model"] == "CTNet":
                model = CTNet(len(labels)).to(device)
            else:

                model = CTNetNoConv(len(labels)).to(device)
        else:
            train_dataset = DatasetMLP(annotations_file=config["dataset"]["train_labels"], feature_dir=config["dataset"]["train_features"] ,labels = labels, exclude=read_file(config["exclude"]))
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

            val_dataset = DatasetMLP(annotations_file=config["dataset"]["val_labels"], feature_dir=config["dataset"]["val_features"],labels = read_file(config["labels"]), exclude=read_file(config["exclude"]))
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

            test_dataset = DatasetMLP(annotations_file=config["dataset"]["test_labels"], feature_dir=config["dataset"]["test_features"],labels = read_file(config["labels"]), exclude=read_file(config["exclude"]))
            test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        
            model = MLP(2048,len(labels)).to(device)
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
        # optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.99, weight_decay=1e-7)
        save_dir = os.path.join(config["save_dir"], run_name)
        # print(val_dataset[0])
        # print(torch.unique(val_dataset[0][0]))
        MLPtrain(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'],save_dir=save_dir, testdataloader=test_loader, testdataset=test_dataset, trainfreq = train_frequencies, valfreq=val_frequencies, testfreq = test_frequencies)
    


    
if __name__ == "__main__":
    config = get_config("config.yaml")

    project_name = config["project_name"]
    # print(len(read_file(config['labels'])))
    run_name = f"AblationNoLungConvmodel_{config['model']}_lr_{config['init_lr']}_bs_{config['batch_size']}_epochs_{config['n_epochs']}_labels_{len(read_file(config['labels']))}connectivity={config['connectivity']}k={config['models'][config['model']]['k']}"
    run_name = run_name + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    wandb.init(
        project=project_name,
        config=config,
        entity="rudranshagarwals",
        name=run_name,
        # mode="disabled"    
    )

    wandb.config.update(config)

    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    save_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(save_dir)

    main(run_name=run_name)

    wandb.finish()