import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch
import torch.nn as nn
import sys
import wandb
import torch.optim as optim

from metrics import evaluate_multilabel_classification
# from torch.utils.data import DataLoader, TensorDataset
from utils import load_checkpoint, save_checkpoint

def MLPtrain(model, traindataloader, criterion, traindataset, optimizer, device, num_epochs, labels, valdataset, valdataloader, num_workers, save_dir, testdataloader, testdataset, trainfreq, valfreq,testfreq):
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()  
    best_val_map = -1
    best_val_map_epoch = -1
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loss = 0
        val_loss = 0
        epoch_outputs = torch.empty((0,len(labels)))
        epoch_targets = torch.empty((0,len(labels)))
        for inputs, targets, image in traindataloader:
            # mean = inputs.mean(dim=2, keepdim=True)[0]
            # std = inputs.std(dim=2, keepdim=True)[0]
            # inputs = (inputs - mean) / (std + 1e-8) 
            # print(inputs)
            # print(inputs.shape)
            # inputs = inputs.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # edges = edges.to(device)
            optimizer.zero_grad()
            # outputs = model(inputs, edges)
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
            epoch_targets = torch.cat((epoch_targets, targets.cpu()))
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(traindataset)
            train_loss = epoch_loss
        probabilities = torch.sigmoid(epoch_outputs)
        predictions = (probabilities > 0.5).float()

        predictions_np = predictions.detach().numpy()
        targets_np = epoch_targets.numpy()
        
        print(f"Train metrics for epoch {epoch}:\n")
        train_accuracy, train_map, train_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(), labels, trainfreq, len(traindataset))
        print(f'Train Loss: {train_loss:.4f}\n')
        model.eval()

        with torch.no_grad():
            running_loss = 0.0
            epoch_outputs = torch.empty((0,len(labels)))
            epoch_targets = torch.empty((0,len(labels)))
            for inputs, targets, image in valdataloader:
                # mean = inputs.mean(dim=2, keepdim=True)[0]
                # std = inputs.std(dim=2, keepdim=True)[0]
                # inputs = (inputs - mean) / (std + 1e-8) 
                # print(inputs)
                # print(inputs.shape)
                inputs = inputs.to(device)
                # batchx = batchx.to(device)
                targets = targets.to(device)
                # edges = edges.to(device)
                # outputs = model(inputs, edges)
                outputs = model(inputs)
                # print(outputs)
                loss = criterion(outputs, targets)
                epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
                epoch_targets = torch.cat((epoch_targets, targets.cpu()))
                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(valdataset)
                val_loss = epoch_loss
            probabilities = torch.sigmoid(epoch_outputs)
            predictions = (probabilities > 0.5).float()

            predictions_np = predictions.detach().numpy()
            targets_np = epoch_targets.numpy()
            print(f"Val metrics for epoch {epoch}:\n")
            val_accuracy, val_map, val_auc = evaluate_multilabel_classification(targets_np, predictions_np , probabilities.detach().numpy(),labels, valfreq, len(valdataset))
            print(f'Val Loss: {val_loss:.4f}\n')
        
            if val_map > best_val_map:
                best_val_map = val_map
                best_val_map_epoch =  epoch + 1
                save_checkpoint(model, optimizer, save_dir, epoch+1)
    
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy, "train_map": train_map, "val_map": val_map, "train_auc": train_auc, "val_auc":val_auc, "epoch": epoch+1, "best_val_map": best_val_map, "best_val_map_epoch": best_val_map_epoch})
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"{save_dir}/model_{best_val_map_epoch}.pth", device)
    print(f"Loaded best checkpoint from epoch {best_val_map_epoch}")


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

        predictions_np = predictions.detach().numpy()
        targets_np = epoch_targets.numpy()
        print("Test metrics:\n")
        test_accuracy, test_map, test_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(),labels, testfreq, len(testdataset))
        print(f'Test Loss: {test_loss:.4f}\n')
    
        

