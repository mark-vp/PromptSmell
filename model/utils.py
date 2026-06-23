from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, precision_recall_curve, auc)
from pre_data import classes
import matplotlib.pyplot as plt
import seaborn as sns
import os


def eval_results(y_true, y_pred, y_probs_positive):
    accuracy = accuracy_score(y_true, y_pred)
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    report = classification_report(y_true, y_pred, target_names=classes)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, weighted_precision, weighted_recall, weighted_f1, report, cm

def show_confusion_matrix(cm, save_path=None):
    if save_path is not None:
        base_name = os.path.basename(save_path)
    else:
        base_name = '1'
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{base_name} \t Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
def show_pr(precision, recall, pr_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()
