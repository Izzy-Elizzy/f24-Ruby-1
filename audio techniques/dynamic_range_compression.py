# dynamic_range_compression.py
import librosa
import numpy as np
 
def apply_dynamic_range_compression(audio_data, sample_rate, threshold=-20, ratio=4):
    """
    Applies dynamic range compression to audio.
    
    Parameters:
    audio_data (np.ndarray): The input audio signal.
    sample_rate (int): The sampling rate of the audio.
    threshold (int): The threshold in decibels (dB) above which compression is applied (default: -20 dB).
    ratio (int): The compression ratio applied to the signal above the threshold. (default: 4)

    Returns:
    compressed_audio (np.ndarray): The audio signal after dynamic range compression.

    """
    audio_db = librosa.amplitude_to_db(np.abs(audio_data))
    compressed_audio_db = np.where(audio_db > threshold, threshold + (audio_db - threshold) / ratio, audio_db)
    compressed_audio = librosa.db_to_amplitude(compressed_audio_db) * np.sign(audio_data)
    return compressed_audio