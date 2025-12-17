import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "tf_events/entropy_scaling_tf_events"



VALUES = {
    "-1.0": os.path.join(BASE, "-1.0"),
    "-0.5": os.path.join(BASE, "-0.5"),
    "-0.3": os.path.join(BASE, "-0.3"),
    "0.0 (Baseline)": os.path.join(BASE, "0"),
    "0.5": os.path.join(BASE, "0.5"),
    "1.0": os.path.join(BASE, "1"),
}


plot_comparison(VALUES, "Entropy Scaling")

