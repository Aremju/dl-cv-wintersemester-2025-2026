import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "tf_events_final"

EVENT_FILE_DICT = {
    "Baseline": os.path.join(BASE, "entropy_0.0_temp_1.0_heads_0_epochs_5(baseline)"),
    "Entropy: -0.3, Temp: 1.3, Heads: 2": os.path.join(BASE, "entropy_-0.3_temp_1.3_heads_2_epochs_5"),
    "Entropy: -0.5, Temp: 1.5, Heads: 3": os.path.join(BASE, "entropy_-0.5_temp_1.5_heads_3_epochs_5"),
    "Entropy: 0.3, Temp: 1.3, Heads: 2": os.path.join(BASE, "entropy_0.3_temp_1.3_heads_2_epochs_5"),
}

plot_comparison(EVENT_FILE_DICT, "Final_TF_Events_Combinations")

