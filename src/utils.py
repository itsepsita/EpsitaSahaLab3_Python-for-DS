import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
def plot_roc_curve(fpr, tpr, auc):
    """
    Plots the ROC curve for the given false positive rate, true positive rate, and AUC.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
