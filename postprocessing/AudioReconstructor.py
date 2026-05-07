import numpy as np
import librosa

class AudioReconstructor:
    def __init__(self, n_fft=512, hop_length=256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def reconstruct(self, clean_chunks, original_phase, original_time_frames):
        clean_chunks = np.squeeze(clean_chunks, axis=-1)
        
        stitched_mag = np.concatenate(clean_chunks, axis=1)
        
        cropped_mag = stitched_mag[:, :original_time_frames]
        
        mag_db = (cropped_mag * 80) - 80 
        
        mag_amp = librosa.db_to_amplitude(mag_db)
        
        complex_stft = mag_amp * original_phase 
        
        clean_wave = librosa.istft(complex_stft, hop_length=self.hop_length, n_fft=self.n_fft)
        
        return clean_wave       