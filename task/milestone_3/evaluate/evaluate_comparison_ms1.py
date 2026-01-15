import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "Comparison_of_MS1"



VALUES = {
    "Baseline": os.path.join(BASE, "Baseline"),
    "BEiT": os.path.join(BASE, "BeiT"),
    "DeiT": os.path.join(BASE, "DeiT"),
}


plot_comparison(VALUES, "Comparison_of_MS1")

