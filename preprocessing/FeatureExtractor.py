import numpy as np
import librosa
import os

from preprocessing.HandleAudio import HandleAudio




class FeatureExtractor:
    def __init__(self):
       self.handle_audio = HandleAudio()
        
    def extract(self, file_path):
        if not os.path.exists(file_path):
            print("File doesn't exist!")
            return
        if not file_path.endswith('.wav'):
            print("Invalid file format!")
            return
        
        n_fft = 512
        hop_length = 256
    
        audio_wave = self.handle_audio.load_audio_file(file_path)
    
        stft_matrix = librosa.stft(audio_wave, n_fft=n_fft, hop_length=hop_length)
        
        magnitude, phase = librosa.magphase(stft_matrix)
        
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        normalized_mag = FeatureExtractor.normalize_spectogram(magnitude_db)

        return phase, normalized_mag
    
    
    
    @staticmethod
    def normalize_spectogram(mag_db):
        normalized_mag = (mag_db + 80) / 80
        
        normalized_mag = np.clip(normalized_mag, 0, 1)
        
        return normalized_mag
    
    def generate_chunks(self, mag_normalized, chunk_width=256):
        _, time_frames = mag_normalized.shape
        chunks = []
        
        if time_frames < chunk_width:
            pad_amount = chunk_width - time_frames
            mag_normalized = np.pad(mag_normalized, ((0,0), (0, pad_amount)), mode='constant', constant_values=0)
            time_frames = chunk_width  
        
        for i in range(0, time_frames - chunk_width + 1, chunk_width):
            chunk = mag_normalized[:, i : i + chunk_width]

            chunk = np.expand_dims(chunk, axis=-1)
            
            chunks.append(chunk)
            
        return chunks