import librosa
import numpy as np
import soundfile as sf
import os

class HandleAudio:
    def load_audio_file(self, audio_file_path, sr=16000):      
        y, _ = librosa.load(audio_file_path, sr=sr)            
        return y
    
    def crop_noise(self, noise_wave, size): 
        noise_length = len(noise_wave)
        
        if noise_length < size:
            repeats = (size // noise_length) + 2
            noise_wave = np.tile(noise_wave, repeats)
            noise_length = len(noise_wave)
        
        
        max_start_index = noise_length - size
        
        if max_start_index == 0:
            start_index = 0
        else:
            start_index = np.random.randint(0, max_start_index)
           
        return noise_wave[start_index : start_index+size]
    
    def mix_audio_with_noise(self, clean_wave, noise_wave, target_snr_db):
        clean_power = np.mean(clean_wave ** 2)
        noise_power = np.mean(noise_wave ** 2)
        
        eps = np.finfo(np.float32).eps
        
        if clean_power < eps or noise_power < eps:
            return clean_wave
                
        snr_linear = 10 ** (target_snr_db / 10)
        noise_scaling_factor = np.sqrt(clean_power / (noise_power * snr_linear))
        
        scaled_noise = noise_wave * noise_scaling_factor
        
        mixed_wave = clean_wave + scaled_noise
        
        max_amplitude = np.max(np.abs(mixed_wave))
        
        if max_amplitude > 1.0:
            mixed_wave = mixed_wave / max_amplitude
            
        return mixed_wave
    
    def save_audio_file(self, path, file_name, wave, sr=16000):
        if not os.path.exists(path):
            print("Path doesn't exist!")
            return 
        if not file_name.endswith('.wav'):
            print("Extention Invalid!")
            return 
        
        sf.write(os.path.join(path, file_name), wave, sr)
        print(f"{file_name} saved successfully!")
        
    