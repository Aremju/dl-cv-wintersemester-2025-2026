import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "classification_report.csv"

def plot_f1_scores(csv_path):
    """Plots the F1-Scores of the classification report as a bar chart."""
    df = pd.read_csv(csv_path)

    df.rename(columns={df.columns[0]: "class"}, inplace=True)
    class_df = df[~df["class"].isin(["accuracy", "macro avg", "weighted avg"])]

    classes = class_df["class"]
    f1_scores = class_df["f1-score"].astype(float)

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
    plt.ylabel("F1-Score")
    plt.xlabel("Klasse")
    plt.ylim(0.85, 1.0)

    plt.title("F1-Scores pro Klasse")

    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(os.path.join("graphs", "f1_scores.png"))
    plt.close()

plot_f1_scores(csv_path)
