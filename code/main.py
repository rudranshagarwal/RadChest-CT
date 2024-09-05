import os
import wandb
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


from train.GCNtrain import GCNtrain
from model.MLP import MLP
from model.GCN import GCN
from model.GAT import GAT


from dataset.DatasetGCN import CustomDataset as DatasetGCN 
from dataset.DatasetMLP import CustomDataset as DatasetMLP
from utils import get_config, set_seed, read_file

# from train import train
# from test import test





def main(run_name):
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_file(config["labels"])
    if config["model"][0] == 'G':
        train_dataset = DatasetGCN(annotations_file=config["dataset"]["train_labels"], feature_dir=config["dataset"]["train_features"], slic_dir=config["dataset"]["train_slic"], lungmask_dir=config["dataset"]["train_lungmask"], centroids_dir=config["dataset"]["train_centroids"], k=config["models"][config['model']]["k"] ,labels = labels, exclude=read_file(config["exclude"]))
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

        val_dataset = DatasetGCN(annotations_file=config["dataset"]["val_labels"], feature_dir=config["dataset"]["val_features"], slic_dir=config["dataset"]["val_slic"], lungmask_dir=config["dataset"]["val_lungmask"], centroids_dir=config["dataset"]["val_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]))
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        test_dataset = DatasetGCN(annotations_file=config["dataset"]["test_labels"], feature_dir=config["dataset"]["test_features"], slic_dir=config["dataset"]["test_slic"], lungmask_dir=config["dataset"]["test_lungmask"], centroids_dir=config["dataset"]["test_centroids"], k=config["models"][config['model']]["k"] ,labels = read_file(config["labels"]), exclude=read_file(config["exclude"]))
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        if(config["model"] == "GCN"):
            model = GCN(len(labels)).to(device)
        elif(config["model"] == "GAT"):
            model = GAT(len(labels), config["models"]["GAT"]["num_heads"]).to(device)
        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])

        save_dir = os.path.join(config["save_dir"], run_name)
        GCNtrain(model=model, traindataloader=train_loader, criterion=criterion, traindataset=train_dataset, optimizer=optimizer, device=device, num_epochs=config['n_epochs'],labels=labels, valdataset= val_dataset, valdataloader=val_loader, num_workers=config['num_workers'],save_dir=save_dir, testdataloader=test_loader, testdataset=test_dataset)
    
    # load_best_checkpoint
    # test(test_loader=test_loader, model, criterion, device=device)


    
if __name__ == "__main__":
    config = get_config("config.yaml")

    project_name = config["project_name"]
    # print(len(read_file(config['labels'])))
    run_name = f"model_{config['model']}_lr_{config['init_lr']}_bs_{config['batch_size']}_epochs_{config['n_epochs']}_labels_{len(read_file(config['labels']))}"
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