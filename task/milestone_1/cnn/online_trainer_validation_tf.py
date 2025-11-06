import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import argparse
import datetime

# ----------------------------- CONFIG -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="../../data/animal_images", help="root folder of dataset")
parser.add_argument("--num-epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--class-num", type=int, default=15)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--net-name", type=str, default="efficientnet-b4")
opt = parser.parse_args()

data_dir = opt.data_dir
num_epochs = opt.num_epochs
batch_size = opt.batch_size
class_num = opt.class_num
lr = opt.lr
momentum = opt.momentum
net_name = opt.net_name

print(f"Dataset: {data_dir} | Epochs: {num_epochs} | Batch size: {batch_size} | Classes: {class_num}")

# ----------------------------- DATA LOADING -----------------------------
def load_data(data_dir, input_size=(380, 380), batch_size=32):
    train_dir = os.path.join(data_dir, "Training Data", "Training Data")
    val_dir = os.path.join(data_dir, "Validation Data", "Validation Data")

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=input_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=input_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Data normalization (same as ImageNet)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds

train_ds, val_ds = load_data(data_dir)

# ----------------------------- MODEL -----------------------------
def build_model(class_num):
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(380, 380, 3),
        pooling="avg"
    )
    base_model.trainable = True  # Fine-tuning

    model = models.Sequential([
        base_model,
        layers.Dropout(0.4),
        layers.Dense(class_num, activation="softmax")
    ])

    return model

model = build_model(class_num)
model.summary()

# ----------------------------- TRAINING SETUP -----------------------------
optimizer = optimizers.SGD(learning_rate=lr, momentum=momentum)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# ----------------------------- CALLBACKS -----------------------------
log_dir = os.path.join(data_dir, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_dir = os.path.join(data_dir, "model")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, f"{net_name}.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: lr * (0.8 ** (epoch // 10)), verbose=1
)

# ----------------------------- TRAINING -----------------------------
history = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=val_ds,
    callbacks=[tensorboard_callback, checkpoint_callback, lr_scheduler]
)

# ----------------------------- EVALUATION -----------------------------
print("\n✅ Testing best model...")
model.load_weights(os.path.join(checkpoint_dir, f"{net_name}.h5"))
loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.4f}")
