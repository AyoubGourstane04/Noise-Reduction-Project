import librosa
import os
import numpy as np
import tensorflow as tf

from preprocessing.FeatureExtractor import FeatureExtractor



class DataGenerator:
    def __init__(self, clean_dir, noise_dir):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.extractor = FeatureExtractor()
        self.file_names = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
        print(f"\n--- GENERATOR INIT ---")
        print(f"Looking in: {clean_dir}")
        print(f"Found {len(self.file_names)} audio files.")
        print(f"----------------------\n")
    
    def generate_pairs(self):
        for file_name in self.file_names:
            print(f"Extracting features for: {file_name}...")
            clean_path = os.path.join(self.clean_dir, file_name)
            noise_path = os.path.join(self.noise_dir, file_name)
            
            _, clean_mag = self.extractor.extract(clean_path)
            _, noise_mag = self.extractor.extract(noise_path)
            
            clean_chunks = self.extractor.generate_chunks(clean_mag)
            noise_chunks = self.extractor.generate_chunks(noise_mag)
            
            for n_chuck, c_chunk in zip(noise_chunks, clean_chunks):
                yield (n_chuck, c_chunk)
    
    def get_tf_dataset(self, batch_size=16):
        output_signature = (
            tf.TensorSpec(shape=(257, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(257, 256, 1), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            self.generate_pairs,
            output_signature=output_signature
        )
        
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(batch_size)
        dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    
    
    

