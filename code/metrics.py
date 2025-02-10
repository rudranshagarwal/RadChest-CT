import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_recall_fscore_support
import wandb
def evaluate_multilabel_classification(y_true, y_pred, probabilities, labels, frequencies, size):
    """
    Evaluates metrics for a multilabel classification task.
    
    Parameters:
    - y_true (np.ndarray): Original true labels (binary or continuous values).
    - y_pred (np.ndarray): Predicted labels (binary or continuous values).

    Prints:
    - Average accuracy
    - Mean Average Precision (MAP)
    - Area Under ROC Curve (AUC)
    - Accuracy, Precision, and Recall for each label
    """
    avg_accuracy = np.mean(np.equal(y_true, y_pred).astype(float))
    
    map_score = average_precision_score(y_true, probabilities, average='macro')
    map_scores = average_precision_score(y_true, probabilities, average=None)
    auc_score = roc_auc_score(y_true, probabilities, average='macro', multi_class='ovr')
    auc_scorew = roc_auc_score(y_true, probabilities, average='weighted', multi_class='ovr')
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    aur_scores = roc_auc_score(y_true, probabilities, average=None)
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print(f"Area Under ROC Curve Weighted (AUC): {auc_scorew:.4f}")
    frequencies = list(frequencies.values())
    # print(frequencies)
    for i in range(y_true.shape[1]):
        print(f"Label {labels[i]}:")
        # print(f"  Accuracy: {accuracy_score(y_true[:, i], y_pred[:, i]):.4f}")
        print(f"  Precision: {map_scores[i]:.4f}")
        # print(f"  Recall: {recall[i]:.4f}")
        # print(f"  F1: {f1[i]:.4f}")
        print(f"  AUR score: {aur_scores[i]:.4f}")
        print(f"  Percentage: {frequencies[i]/size:.4f}")
    


    # new_y_true = y_true[(y_true[:, 3] == 1 )& (y_true[:, 7] == 1)]
    # new_probabilities = probabilities[(y_true[:, 3] == 1) & (y_true[:, 7] == 1)]
    # print(new_y_true)
    # new_aur_score = roc_auc_score(new_y_true, new_probabilities, average=None)
    # new_map_score = average_precision_score(new_y_true, new_probabilities, average=None)
    # print("Reticulation = 1 ILD = 1")
    # print(f"  Precision: {new_map_score[3]:.4f} {new_map_score[7]:.4f}")
        
    # print(f"  AUR score: {aur_scores[3]:.4f} {aur_scores[7]:.4f}")


    # new_y_true = y_true[(y_true[:, 3] == 0) & (y_true[:, 7] == 1)]
    # new_probabilities = probabilities[(y_true[:, 3] == 0) & (y_true[:, 7] == 1)]
    
    # new_aur_score = roc_auc_score(new_y_true, new_probabilities, average=None)
    # new_map_score = average_precision_score(new_y_true, new_probabilities, average=None)
    # print("RETICULATION = 0 ILD = 1")
    # print(f"  Precision: {new_map_score[3]:.4f} {new_map_score[7]:.4f}")
        
    # print(f"  AUR score: {aur_scores[3]:.4f} {aur_scores[7]:.4f}")

    return avg_accuracy, map_score, auc_score



