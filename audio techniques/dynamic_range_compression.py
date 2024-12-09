import librosa
import numpy as np
 
def apply_dynamic_range_compression(audio_data, sample_rate, threshold=-20, ratio=4):
    """Applies dynamic range compression to audio."""
    audio_db = librosa.amplitude_to_db(np.abs(audio_data))
    compressed_audio_db = np.where(audio_db > threshold, threshold + (audio_db - threshold) / ratio, audio_db)
    compressed_audio = librosa.db_to_amplitude(compressed_audio_db) * np.sign(audio_data)
    return compressed_audio