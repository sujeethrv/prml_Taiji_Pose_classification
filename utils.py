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

def plot_confusion_matrix(true_labels, predicted_labels, class_names, output_file):
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_df, annot=True, fmt="d")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_file)

def save_predictions(true_labels, predicted_labels, output_file):
    pred_df = pd.DataFrame({"true_labels": true_labels, "predicted_labels": predicted_labels})
    pred_df.to_csv(output_file, index=False)
