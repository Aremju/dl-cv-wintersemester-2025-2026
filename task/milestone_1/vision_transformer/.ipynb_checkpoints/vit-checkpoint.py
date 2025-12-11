import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
import glob, random, os, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model_name = "google/vit-base-patch16-224"
batch_size = 32
image_size = 224
classes = 15
epochs = 5
train_path = "../../../data/animal_images/Training Data/Training Data/"
val_path = "../../../data/animal_images/Validation Data/Validation Data/"
test_path = "../../../data/animal_images/Testing Data/Testing Data/"


# loads all required data for fine-tuning and evalutation
def load_data():
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        label_mode='categorical',
        image_size=(image_size, image_size),
        shuffle=True,
        seed=1,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        label_mode='categorical',
        image_size=(image_size, image_size),
        shuffle=False,
        seed=1,
    )

    """ test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        label_mode='categorical',
        image_size=(image_size, image_size),
        shuffle=True,
        seed=1
    ) """
    

    return train_ds, val_ds

def normalize_data(train_ds, val_ds):
    # Rescaling layer
    rescale = tf.keras.layers.Rescaling(1./255)

    # Adapts normalization layer to training data. Not adapted to validation and test to avoid data leakage!
    """with tf.device('/CPU:0'):
        image_batches = train_ds.map(lambda x, y: x)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(image_batches)
    
    # Applying both transformation layers
    normalized_train_ds = train_ds.map(lambda x, y: (normalizer(rescale(x)), y))
    normalized_val_ds   = val_ds.map(lambda x, y: (normalizer(rescale(x)), y))"""

    normalized_train_ds = train_ds.map(lambda x, y: (rescale(x)), y)
    normalized_val_ds   = val_ds.map(lambda x, y: (rescale(x)), y)

def preprocess(image, label):
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.uint8)
    inputs = processor(images=image.numpy(), return_tensors="tf")
    pixel_values = inputs["pixel_values"][0]
    return pixel_values, label

def preprocess_tf(image, label):
    pixel_values, label = tf.py_function(preprocess, [image, label], [tf.float32, tf.int64])
    pixel_values.set_shape((3, image_size, image_size))
    return pixel_values, label

def train(train_ds, val_ds):
    #train_ds = train_ds.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
    #val_ds = val_ds.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)

    #train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')

    """optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )"""

    metric_name = "accuracy"

    # Define Train Parameters
    args = TrainingArguments(
        f"breed-classification",
        use_cpu = False,
        eval_strategy="steps",
        logging_steps = 100,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs',
        remove_unused_columns=False,
    )
    
    # Train
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()


if __name__ == "__main__":
    train_ds, val_ds = load_data()
    
    # normalized_train, normalized_val = normalize_data(train, val)
    train(train_ds, val_ds)