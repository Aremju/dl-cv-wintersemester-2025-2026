import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "final_classification_report.csv"

def plot_precision_values_per_class(csv_path):
    """Plots the Precision-Values of the classification report as a bar chart for each class."""
    df = pd.read_csv(csv_path)

    df.rename(columns={df.columns[0]: "class"}, inplace=True)
    class_df = df[~df["class"].isin(["accuracy", "macro avg", "weighted avg"])]

    classes = class_df["class"]
    f1_scores = class_df["precision"].astype(float)

    x = np.arange(len(classes))
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    plt.figure(figsize=(12, 6))
    plt.bar(x, f1_scores, color=colors, width=0.7)


    for i, v in enumerate(f1_scores):
        plt.text(
            i,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel("Precision")
    plt.xlabel("Class")
    plt.ylim(0, 1.0)

    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(os.path.join("graphs", "precision_scores.png"))
    plt.close()

plot_precision_values_per_class(csv_path)
