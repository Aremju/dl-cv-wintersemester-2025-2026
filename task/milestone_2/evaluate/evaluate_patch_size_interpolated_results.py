import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "tf_events/patch_size_interpolated_tf_events"


VALUES = {
    "8": os.path.join(BASE, "8"),
    "16 (Baseline)": os.path.join(BASE, "16"),
    "32": os.path.join(BASE, "32"),
    "64": os.path.join(BASE, "64"),
    "224": os.path.join(BASE, "224"),
}

plot_comparison(VALUES, "Patch Sizes Interpolated")

