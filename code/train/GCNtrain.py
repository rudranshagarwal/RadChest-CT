import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch
import torch.nn as nn
import sys
import wandb
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from metrics import evaluate_multilabel_classification
# from torch.utils.data import DataLoader, TensorDataset
from utils import load_checkpoint, save_checkpoint

def GCNtrain(model, traindataloader, criterion, traindataset, optimizer, device, num_epochs, labels, valdataset, valdataloader, num_workers, save_dir, testdataloader, testdataset, trainfreq, valfreq,testfreq):
    def save_model_parameters(model, file_path):
        torch.save(model.state_dict(), file_path)
        print(f"Model parameters saved to {file_path}")
    patience = 10
    patience_counter=0
    # Specify the output file path for saving parameters
    output_file = 'model_parameters_2.pth'
    mean = torch.load('../mean6.pth')
    std = torch.load('../std6.pth')
    # Save the model parameters
    # save_model_parameters(model, output_file)
    
    best_val_aur = -1
    best_val_aur_epoch = -1
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        # running_loss2 = 0.0
        train_loss = 0
        val_loss = 0
        epoch_outputs = torch.empty((0,len(labels)))
        # epoch_outputs2 = torch.empty((0,420, 2048))
        epoch_targets = torch.empty((0,len(labels)))
        for inputs, targets, edges, image, connectivity in traindataloader:
            # mean = inputs.mean(dim=2, keepdim=True)
            # std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-8) 
            # inputs = (inputs - mean)
            # print(inputs.dtype)
            graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
            dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
            batchx = []
            for databatch in dataloader_graph:
                batchx = databatch
            # print(inputs.shape)
            # inputs = inputs.to(device)
            batchx = batchx.to(device)
            targets = targets.to(device)
            # laplacian = laplacian.to(device)
            # edges = edges.to(device)
            optimizer.zero_grad()
            # outputs = model(inputs, edges)
            outputs = model(batchx, connectivity)
            # outputs2 = model(batchx)
            # print(outputs)
            # print(targets)
            loss = criterion(outputs, targets)
            # loss2 = criterion(outputs2, targets)
            # print(loss.item())
            loss.backward()
            # loss2.backward()
            optimizer.step()
            epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
            # epoch_outputs2 = torch.cat((epoch_outputs2, inputs.cpu()))
            epoch_targets = torch.cat((epoch_targets, targets.cpu()))
            running_loss += loss.item() * inputs.size(0)
            # running_loss2 += loss2.item() * inputs.size(0)
            # break
        # print(running_loss)
        epoch_loss = running_loss / len(traindataset)
        # print(running_loss2/len(traindataset))
        train_loss = epoch_loss
            # break
        # print(epoch_outputs)
        probabilities = torch.sigmoid(epoch_outputs)
        # torch.save(epoch_outputs, './test1.pth')
        # torch.save(epoch_outputs2, './test2.pth')
        # torch.save(epoch_targets, 'targets2.pth')
        predictions = (probabilities > 0.5).float()

        predictions_np = predictions.detach().numpy()
        targets_np = epoch_targets.numpy()

        print(f"Train metrics for epoch {epoch}:\n")
        train_accuracy, train_map, train_auc = evaluate_multilabel_classification(targets_np, predictions_np, probabilities.detach().numpy(), labels, trainfreq, len(traindataset))
        print(f'Train Loss: {train_loss:.4f}\n')

        model.eval()
        flag = 0
        with torch.no_grad():
            running_loss = 0.0
            epoch_outputs = torch.empty((0,len(labels)))
            epoch_targets = torch.empty((0,len(labels)))
            for inputs, targets, edges, image, connectivity in valdataloader:
                # mean = inputs.mean(dim=2, keepdim=True)
                # std = inputs.std(dim=2, keepdim=True)
                inputs = (inputs - mean) / (std + 1e-8) 
                # inputs = (inputs - mean)
                # print(inputs)
                graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
                dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
                batchx = []
                for databatch in dataloader_graph:
                    batchx = databatch
                # print(inputs.shape)
                # inputs = inputs.to(device)
                batchx = batchx.to(device)
                targets = targets.to(device)
                # laplacian = laplacian.to(device)
                # edges = edges.to(device)
                # outputs = model(inputs, edges)
                outputs = model(batchx, connectivity)
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
        
            if val_auc > best_val_aur:
                flag = 1
                best_val_aur = val_auc
                best_val_aur_epoch =  epoch + 1
                save_checkpoint(model, optimizer, save_dir, epoch+1)
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy, "train_map": train_map, "val_map": val_map, "train_auc": train_auc, "val_auc":val_auc, "epoch": epoch+1, "best_val_aur": best_val_aur, "best_val_aur_epoch": best_val_aur_epoch})
        if flag:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch}")
            break
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"{save_dir}/model_{best_val_aur_epoch}.pth", device)
    print(f"Loaded best checkpoint from epoch {best_val_aur_epoch}")

    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        test_loss = 0
        epoch_outputs = torch.empty((0,len(labels)))
        epoch_targets = torch.empty((0,len(labels)))
        for inputs, targets, edges, image,connectivity in testdataloader:
            # mean = inputs.mean(dim=2, keepdim=True)
            # std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-8) 
            # inputs = (inputs - mean)
            # print(inputs)
            graphs = [Data(x = inputs[j], edge_index = edges[j]) for j in range(inputs.shape[0])]
            dataloader_graph = DataLoader(graphs, batch_size=inputs.shape[0],num_workers=num_workers, shuffle=False)
            batchx = []
            for databatch in dataloader_graph:
                batchx = databatch
            # print(inputs.shape)
            # inputs = inputs.to(device)
            batchx = batchx.to(device)
            targets = targets.to(device)
            # laplacian = laplacian.to(device)
            # edges = edges.to(device)
            # outputs = model(inputs, edges)
            outputs = model(batchx, connectivity)
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
    
        

