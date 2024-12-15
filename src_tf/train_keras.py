import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
layers = keras.layers
models = keras.models

log_dir = "logs/fit"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

base_model = tf.keras.applications.MobileNetV2(input_shape=(640, 640, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']
)

train_dir = "dataset/Images/train"
val_dir = "dataset/Images/val"

def load_dataset_with_progress(directory, image_size, batch_size):
    """
    Load dataset from the specified directory and display progress using tqdm.
    """
    print(f"Loading dataset from {directory}...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size
    )
    for _ in tqdm(dataset, desc="Processing dataset"):
        pass
    return dataset

train_dataset = load_dataset_with_progress(train_dir, (640, 640), 32)
val_dataset = load_dataset_with_progress(val_dir, (640, 640), 32)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[tensorboard_callback]
)

print("Converting and saving the model to .tflite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model saved as 'model.tflite'.")
