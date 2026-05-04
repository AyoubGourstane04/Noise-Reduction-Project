import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_clean_path = os.path.join(BASE_DIR, "../data/clean_audio/wavs")
train_noisy_path = os.path.join(BASE_DIR, "../data/dataset")

val_clean_path = os.path.join(BASE_DIR, "../data/val_clean")
val_noisy_path = os.path.join(BASE_DIR, "../data/val_noisy")


all_files = os.listdir(train_clean_path)
unique_speakers = list(set([f.split('_')[0] for f in all_files if f.endswith('.wav')]))


num_val_speakers = int(len(unique_speakers) * 0.10)
val_speakers = random.sample(unique_speakers, num_val_speakers)

print(f"Selected {len(val_speakers)} speakers for validation: {val_speakers}")

moved_count = 0
for file_name in all_files:
    speaker = file_name.split('_')[0]
    if speaker in val_speakers:
        shutil.move(
            os.path.join(train_clean_path, file_name),
            os.path.join(val_clean_path, file_name)
        )
        shutil.move(
            os.path.join(train_noisy_path, file_name),
            os.path.join(val_noisy_path, file_name)
        )
        
        moved_count+=1
print(f"Successfully moved {moved_count} pairs of files to the validation folders!")