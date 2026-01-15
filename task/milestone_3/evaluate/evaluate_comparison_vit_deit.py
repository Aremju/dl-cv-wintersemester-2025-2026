import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "Comparison of ViT and DeiT"



VALUES = {
    "ViT": os.path.join(BASE, "ViT"),
    "DeiT": os.path.join(BASE, "DeiT"),
}


plot_comparison(VALUES, "Comparison of ViT and DeiT")

