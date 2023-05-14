import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_metrics(true_labels, predicted_labels, class_names, output_file):
    report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_file)
    return report_df

def plot_confusion_matrix(y_true, y_pred, class_names, output_file):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix using seaborn heatmap without numbers
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cmap='coolwarm', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Normalized Frequency'})
    
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    plt.savefig(output_file)
    plt.show()


def save_predictions(true_labels, predicted_labels, output_file):
    pred_df = pd.DataFrame({"true_labels": true_labels, "predicted_labels": predicted_labels})
    pred_df.to_csv(output_file, index=False)
