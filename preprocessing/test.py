import os
import numpy as np
from preprocessing.HandleAudio import HandleAudio
from preprocessing.DataGenerator import DataGenerator


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# handle_audio_file = HandleAudio()


# audio_files_path = os.path.join(BASE_DIR, "../data/clean_audio/wavs")
# clean_audio_files = os.listdir(audio_files_path)

# clean_audio_files_paths = [os.path.join(audio_files_path, file_path) for file_path in clean_audio_files]
    


# noise_files_path = os.path.join(BASE_DIR, "../data/noise/")
# noise_audio_folders = os.listdir(noise_files_path)

# noise_audio_folders_paths = [
#                             os.path.join(noise_files_path, folder) 
#                              for folder in noise_audio_folders 
#                                 if folder.__contains__('fold')
#                             ]


# noise_audio_files_paths = [
#                         os.path.join(folder_path, file_path) 
#                          for folder_path in noise_audio_folders_paths
#                             for file_path in os.listdir(folder_path) 
#                         ]


# print(f'clean_audio_path: {clean_audio_files_paths[0]}')
# print(f'noise_audio_path : {noise_audio_files_paths[0]}')
# print("---------------------------------------")


# clean_wave = handle_audio_file.load_audio_file(clean_audio_files_paths[0])
# noise_wave = handle_audio_file.load_audio_file(noise_audio_files_paths[0])

# print(f'clean_wave_shape : {clean_wave.shape}')
# print(f'noise_wave_shape : {noise_wave.shape}')
# print("---------------------------------------")

# clean_length = len(clean_wave)

# noise_wave = handle_audio_file.crop_noise(noise_wave, clean_length)

# print(f'clean_wave_shape : {clean_wave.shape}')
# print(f'noise_wave_shape : {noise_wave.shape}')
# print("---------------------------------------")


# random_snr = np.random.uniform(-5, 15)

# noisy_audio_wave = handle_audio_file.mix_audio_with_noise(clean_wave, noise_wave, random_snr)

# handle_audio_file.save_audio_file(os.path.join(BASE_DIR, "../test"), "mixed_audio.wav", noisy_audio_wave)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(BASE_DIR, "../data/clean_audio/wavs")
noisy_dir = os.path.join(BASE_DIR, "../data/dataset")

data_gen = DataGenerator(clean_dir, noisy_dir)
dataset = data_gen.get_tf_dataset(batch_size=16)

for noisy_batch, clean_batch in dataset.take(1):
    print("Pipeline Success!")
    print(f"Noisy Batch Shape: {noisy_batch.shape}")
    print(f"Clean Batch Shape: {clean_batch.shape}")