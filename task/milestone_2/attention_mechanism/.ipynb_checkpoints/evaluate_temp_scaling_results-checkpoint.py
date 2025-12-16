import os

from task.milestone_2.tf_event_evaluation import plot_comparison

BASE = "tf_events/temp_scaling_tf_events"

TEMPERATURES_TESTED = {
    "0.3": os.path.join(BASE, "0.3"),
    "0.5": os.path.join(BASE, "0.5"),
    "1.0": os.path.join(BASE, "1.0"),
    "1.3": os.path.join(BASE, "1.3"),
    "1.5": os.path.join(BASE, "1.5"),
    "2.0": os.path.join(BASE, "2.0"),
}

VAL_ACC = "eval/accuracy"
VAL_LOSS = "eval/loss"

plot_comparison(TEMPERATURES_TESTED, "Temperature Scaling")
