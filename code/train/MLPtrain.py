import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../')
device = "cuda" if torch.cuda.is_available() else "cpu"


def MLPtrain(num_features, num_labels):

    model = MultiLabelNN(num_features, num_labels).to(device)

    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    num_epochs = 10
    dataloader = DataLoader(traindata, batch_size=512, shuffle=True)
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        running_accuracy = 0
        epoch_outputs = torch.empty((0,6))
        epoch_targets = torch.empty((0,6))
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(3)
            optimizer.zero_grad()
            outputs = model(torch.mean(inputs, dim=1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_outputs = torch.cat((epoch_outputs, outputs.cpu()))
            epoch_targets = torch.cat((epoch_targets, targets.cpu()))
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(traindata)
        probabilities = torch.sigmoid(epoch_outputs)
        predictions = (probabilities > 0.5).float()

        predictions_np = predictions.detach().numpy()
        targets_np = epoch_targets.numpy()

        average_precision = average_precision_score(targets_np, predictions_np, average='macro')
        auroc_score = roc_auc_score(targets_np,predictions_np,average='macro')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Average Precision (Macro): {average_precision:.4f}, AUROC: {auroc_score:.4f}')
