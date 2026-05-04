import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocessing.DataGenerator import DataGenerator 
from training.model import build_unet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_clean_path = os.path.join(BASE_DIR, "../data/clean_audio/wavs")
train_noisy_path = os.path.join(BASE_DIR, "../data/dataset")

val_clean_path = os.path.join(BASE_DIR, "../data/val_clean")
val_noisy_path = os.path.join(BASE_DIR, "../data/val_noisy")



print("Initializing Data Generator...")
batch_size = 32

train_gen = DataGenerator(train_clean_path, train_noisy_path)
train_dataset = train_gen.get_tf_dataset(batch_size=batch_size)

val_gen = DataGenerator(val_clean_path, val_noisy_path)
val_dataset = val_gen.get_tf_dataset(batch_size=batch_size)

print("Building U-Net Model")
model = build_unet()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')

checkpoint_path = os.path.join(BASE_DIR, "../models/unet_best_weights.keras")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='loss', mode='min', verbose=1),
    EarlyStopping(monitor='loss', patience=3, verbose=1)
]

print('Starting Training Loop...')
EPOCHS = 50

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Training completed successfully!")