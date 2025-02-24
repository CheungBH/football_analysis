import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_confusion_matrix(predictions_labels):
    """
    Generate a confusion matrix from a list of predicted and actual labels.

    :param predictions_labels: List of lists, where each sublist contains [predicted_label, actual_label]
    :return: Pandas DataFrame representing the confusion matrix
    """
    # Extract unique labels
    labels = sorted(set(label for pair in predictions_labels for label in pair))

    # Create a label index mapping
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Initialize confusion matrix
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Populate the confusion matrix
    for actual, predicted in predictions_labels:
        matrix[label_to_index[actual], label_to_index[predicted]] += 1  # Row: actual, Column: predicted

    # Convert to DataFrame for better readability
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "Actual"
    df.columns.name = "Predicted"

    return df


def plot_confusion_matrix(conf_matrix, save_path="confusion_matrix.png"):
    """
    Plot the confusion matrix using seaborn heatmap.

    :param conf_matrix: Pandas DataFrame representing the confusion matrix
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=1, linecolor="black")

    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

