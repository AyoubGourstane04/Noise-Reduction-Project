import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import tensorflow as tf
import numpy as np
import soundfile as sf

from postprocessing.AudioReconstructor import AudioReconstructor
from preprocessing.FeatureExtractor import FeatureExtractor



@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "../models/unet_best_weights.keras")
    return tf.keras.models.load_model(model_path)


def predict(noisy_path, clean_destination_path, model):
    
    extractor = FeatureExtractor()
    print("Extracting features...")
    phase, mag_norm = extractor.extract(noisy_path)
    
    original_length = mag_norm.shape[1]
    
    chunks = extractor.generate_chunks(mag_norm)
    chunk_array = np.array(chunks)
    
    print("U-Net is cleaning the audio...")
    clean_chunks = model.predict(chunk_array)
    
    print("--- DIAGNOSTIC CHECK ---")
    print("Max value predicted:", np.max(clean_chunks))
    print("Mean value predicted:", np.mean(clean_chunks))
    print("------------------------")
        
    print("Reconstructing audio wave...")
    reconstructor = AudioReconstructor()
    
    clean_wave = reconstructor.reconstruct(
        clean_chunks=clean_chunks, 
        original_phase=phase, 
        original_time_frames=original_length
    )
    
    final_clean_wave = clean_wave / np.max(np.abs(clean_wave))
    
    sf.write(clean_destination_path, final_clean_wave, 16000)
    print(f"Success! Cleaned audio saved to: {clean_destination_path}")


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOADS_DIR = os.path.join(BASE_DIR, "../uploads")
    
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    st.title("🎙️ AI Audio Noise Reduction")
    st.markdown("Upload a noisy `.wav` file, and the U-Net will isolate the human voice.")

    model = load_model()
    
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    
    if uploaded_file is not None:
        st.subheader("Original Noisy Audio")
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Clean Audio"):
            with st.spinner('Neural Network is processing...'):
                
                temp_noisy_path = os.path.join(UPLOADS_DIR, "temp_noisy.wav")               
                temp_clean_path = os.path.join(UPLOADS_DIR, "temp_clean.wav")
                
                with open(temp_noisy_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                predict(temp_noisy_path, temp_clean_path, model)
                
            st.success('Audio cleaned successfully!')
            
            st.subheader("Cleaned Audio Output")
            st.audio(temp_clean_path, format='audio/wav')
            
            with open(temp_clean_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Cleaned .wav",
                    data=file,
                    file_name="restored_voice.wav",
                    mime="audio/wav"
                )
                
            
            if os.path.exists(temp_noisy_path):
                os.remove(temp_noisy_path)
                
            if os.path.exists(temp_clean_path):
                os.remove(temp_clean_path)
            
            
if __name__ == "__main__":
    main()