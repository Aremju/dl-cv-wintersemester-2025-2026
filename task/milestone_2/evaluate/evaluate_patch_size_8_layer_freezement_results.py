import os

from task.milestone_2.evaluate.tf_event_evaluation import plot_comparison

BASE = "tf_events/patch_size_8_detailed"



VALUES = {
    "Interpolated": os.path.join(BASE, "8_interpolated"),
    "Not Interpolated": os.path.join(BASE, "8_no_interpolation"),
    "Layer Unfreezing" : os.path.join(BASE, "8_layer_unfreezing_2_layers"),
}


plot_comparison(VALUES, "Patch Size 8 experiments")

