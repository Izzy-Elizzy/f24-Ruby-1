# slight_time_stretching.py
import librosa
 
def apply_slight_time_stretching(audio_data, sample_rate, rate=1.01):
    """
    Applies slight time stretching to audio.
    
    Parameters:
    audio_data (np.ndarray): The input audio signal.
    sample_rate (int): The sampling rate of the audio signal.
    rate (float): The rate of time stretching. (default: 1.01)

    Returns:
    time_stretched_audio (np.ndarray): Time stretched audio signal.
    """
    time_stretched_audio = librosa.effects.time_stretch(audio_data, rate=rate)
    return time_stretched_audio