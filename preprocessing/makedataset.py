import os
import numpy as np
from preprocessing.HandleAudio import HandleAudio
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
handle_audio_file = HandleAudio()

clean_dir = os.path.join(BASE_DIR, "../data/clean_audio/wavs")
clean_paths = [os.path.join(clean_dir, file_path) for file_path in os.listdir(clean_dir) if file_path.endswith('.wav')]

noise_dir = os.path.join(BASE_DIR, "../data/noise/")
noise_paths = []
for folder in os.listdir(noise_dir):
    if 'fold' in folder:
        folder_path = os.path.join(noise_dir, folder) 
        noise_paths.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')])
        
output_dir = os.path.join(BASE_DIR, "../dataset")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for clean_path in clean_paths:
    file_name = os.path.basename(clean_path)
    noise_path = random.choice(noise_paths)
    
    clean_wave = handle_audio_file.load_audio_file(clean_path)
    noise_wave = handle_audio_file.load_audio_file(noise_path)
    
    noise_wave_cropped = handle_audio_file.crop_noise(noise_wave, len(clean_wave))

    snr = np.random.uniform(-5, 15)

    mixed_wave = handle_audio_file.mix_audio_with_noise(clean_wave, noise_wave_cropped, snr)
    
    handle_audio_file.save_audio_file(output_dir, file_name, mixed_wave)

print("Dataset Generated Successfully!")