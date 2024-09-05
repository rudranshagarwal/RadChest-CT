import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_recall_fscore_support
import wandb
def evaluate_multilabel_classification(y_true, y_pred):
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
    
    map_score = average_precision_score(y_true, y_pred, average='macro')
    
    auc_score = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    
    for i in range(y_true.shape[1]):
        print(f"Label {i + 1}:")
        print(f"  Accuracy: {accuracy_score(y_true[:, i], y_pred[:, i]):.4f}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
    
    
    return avg_accuracy, map_score, auc_score



