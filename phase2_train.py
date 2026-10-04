import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# PROJECT PATH
# ============================================================

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(HERE, "phase1_index.csv")

print("Project folder:", HERE)
print("CSV file:", CSV_PATH)

# ============================================================
# LOAD PHASE 1 DATA
# ============================================================

if not os.path.exists(CSV_PATH):
    print("ERROR: phase1_index.csv not found!")
    exit()

df = pd.read_csv(CSV_PATH)

print("\n========== DATASET ==========")
print("Total images:", len(df))
print("\nSplit:")
print(df["split"].value_counts())
print("\nLabels:")
print(df["label"].value_counts())

# ============================================================
# LABEL CONVERSION
# normal = 0
# anemic = 1
# ============================================================

label_map = {
    "normal": 0,
    "anemic": 1
}

df["label_id"] = df["label"].map(label_map)

# ============================================================
# TRAIN / VALIDATION DATA
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

# ============================================================
# IMAGE SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ============================================================
# CREATE TRAIN DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_df["path"].values,
        train_df["label_id"].values
    )
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(500)

train_ds = train_ds.batch(BATCH_SIZE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["path"].values,
        val_df["label_id"].values
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    # Flatten
    layers.Flatten(),

    # Dense layer
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    # Output
    # 0 = Normal
    # 1 = Anemic
    layers.Dense(
        1,
        activation="sigmoid"
    )
])

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n========== MODEL ==========")

model.summary()

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n========== TRAINING STARTED ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ============================================================
# FINAL ACCURACY
# ============================================================

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

print("\n========== TRAINING RESULT ==========")

print(
    f"Training Accuracy: {train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy: {val_accuracy * 100:.2f}%"
)

# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    HERE,
    "ml",
    "anemia_model.keras"
)

model.save(model_path)

print("\n======================================")
print("MODEL TRAINING COMPLETED")
print("======================================")
print("Model saved at:")
print(model_path)
print("======================================")

