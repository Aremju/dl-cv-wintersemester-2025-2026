import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "tf_events_final_configuration_10_epochs"


EVENT_FILE_DICT = {
    "Baseline": os.path.join(BASE, "Baseline"),
    "Entropy: 0.3, Temp: 1.3, Heads: 2": os.path.join(BASE, "Our Model Entropy 0.3 Temp 1.3 Heads 2"),
}


plot_comparison(EVENT_FILE_DICT, "Entropy Scaling - Final 10 Epochs", xTicks=[1,2,3,4,5,6,7,8,9,10])

