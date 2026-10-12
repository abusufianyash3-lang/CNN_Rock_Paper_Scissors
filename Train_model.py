import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

DATA_DIR = Path("dataset")
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 123
CLASSES = ["rock", "paper", "scissors"]

for name in CLASSES:
    folder = DATA_DIR / name
    if not folder.exists() or len(list(folder.glob("*.jpg"))) < 20:
        raise SystemExit(
            f"Not enough images for '{name}'. Run collect_data.py and save at least 20 images per class "
            "(100-200+ is recommended)."
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, class_names=CLASSES, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, class_names=CLASSES, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(96, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)
model.save("rps_cnn.keras")
print("Saved model to rps_cnn.keras")
print("Class order:", train_ds.class_names if hasattr(train_ds, "class_names") else CLASSES)
