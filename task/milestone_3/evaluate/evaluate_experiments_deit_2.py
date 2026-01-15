import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "Experiments_DeiT/2"



VALUES = {
    "Baseline": os.path.join(BASE, "Baseline"),
    "Experiment": os.path.join(BASE, "Experiment"),
}


plot_comparison(VALUES, "Second DeiT Experiment")

