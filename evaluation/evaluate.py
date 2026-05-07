import os
import numpy as np
import librosa
import tensorflow as tf
from pystoi import stoi
from tqdm import tqdm 

from preprocessing.FeatureExtractor import FeatureExtractor
from postprocessing.AudioReconstructor import AudioReconstructor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_snr(clean, noisy):
    """Calculates Signal-to-Noise Ratio in dB"""
    noise = clean - noisy
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))

def evaluate_model(model_path, val_clean_dir, val_noisy_dir, num_samples=50):
    print("Loading Model...")
    model = tf.keras.models.load_model(model_path)
    extractor = FeatureExtractor()
    reconstructor = AudioReconstructor()
    
    clean_files = sorted(os.listdir(val_clean_dir))[:num_samples]
    
    total_stoi_noisy = 0
    total_stoi_clean = 0
    total_snr_noisy = 0
    total_snr_clean = 0
    
    print(f"Evaluating {num_samples} files...")
    
    for filename in tqdm(clean_files):
        clean_path = os.path.join(val_clean_dir, filename)
        noisy_path = os.path.join(val_noisy_dir, filename)
        
        if not os.path.exists(noisy_path): continue

        clean_wave, _ = librosa.load(clean_path, sr=16000)
        noisy_wave, _ = librosa.load(noisy_path, sr=16000)

        phase, mag_norm = extractor.extract(noisy_path)
        original_length = mag_norm.shape[1]
        chunks = extractor.generate_chunks(mag_norm)
        
        pred_chunks = model.predict(np.array(chunks), verbose=0)
        
        denoised_wave = reconstructor.reconstruct(
            clean_chunks=pred_chunks, 
            original_phase=phase, 
            original_time_frames=original_length
        )

        min_len = min(len(clean_wave), len(denoised_wave), len(noisy_wave))
        clean_wave = clean_wave[:min_len]
        noisy_wave = noisy_wave[:min_len]
        denoised_wave = denoised_wave[:min_len]
        
        clean_wave = clean_wave / np.max(np.abs(clean_wave))
        denoised_wave = denoised_wave / np.max(np.abs(denoised_wave))


        total_stoi_noisy += stoi(clean_wave, noisy_wave, 16000, extended=False)
        total_stoi_clean += stoi(clean_wave, denoised_wave, 16000, extended=False)
        
        total_snr_noisy += calculate_snr(clean_wave, noisy_wave)
        total_snr_clean += calculate_snr(clean_wave, denoised_wave)

    avg_stoi_in = total_stoi_noisy / num_samples
    avg_stoi_out = total_stoi_clean / num_samples
    avg_snr_in = total_snr_noisy / num_samples
    avg_snr_out = total_snr_clean / num_samples

    print("\n" + "="*30)
    print("FINAL PERFORMANCE REPORT")
    print("="*30)
    print(f"Average STOI (Noisy):  {avg_stoi_in:.4f}")
    print(f"Average STOI (Cleaned): {avg_stoi_out:.4f} (Intelligibility)")
    print("-" * 30)
    print(f"Average SNR (Noisy):   {avg_snr_in:.2f} dB")
    print(f"Average SNR (Cleaned):  {avg_snr_out:.2f} dB")
    print(f"Total SNR Improvement: {avg_snr_out - avg_snr_in:.2f} dB")
    print("="*30)

if __name__ == "__main__":
    MODEL = os.path.join(BASE_DIR, "../models/unet_best_weights.keras")
    VAL_CLEAN = os.path.join(BASE_DIR, "../data/val_clean")
    VAL_NOISY = os.path.join(BASE_DIR, "../data/val_noisy")
    
    evaluate_model(MODEL, VAL_CLEAN, VAL_NOISY, num_samples=30)