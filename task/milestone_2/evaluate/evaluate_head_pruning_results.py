import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "tf_events/head_pruning_tf_events"



HALVES = {
    "Half-Even": os.path.join(BASE, "even"),
    "Half-uneven": os.path.join(BASE, "uneven"),
    "Random-6": os.path.join(BASE, "6"),
}

RANDOM = {
    "Baseline": os.path.join(BASE, "Baseline"),
    "Random-1": os.path.join(BASE, "1"),
    "Random 2": os.path.join(BASE, "2"),
    "Random 3": os.path.join(BASE, "3"),
    "Random 11": os.path.join(BASE, "11"),
}


plot_comparison(HALVES, "Head Pruning - Even and Uneven")
plot_comparison(RANDOM, "Head Pruning - Random")

