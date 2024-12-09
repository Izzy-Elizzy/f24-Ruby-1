# slight_time_stretching.py
import librosa
 
def apply_slight_time_stretching(audio_data, sample_rate, rate=1.01):
    """Applies slight time stretching to audio."""
    time_stretched_audio = librosa.effects.time_stretch(audio_data, rate=rate)
    return time_stretched_audio