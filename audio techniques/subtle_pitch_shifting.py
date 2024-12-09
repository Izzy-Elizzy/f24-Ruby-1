# subtle_pitch_shifting.py
import librosa
 
def apply_subtle_pitch_shifting(audio_data, sample_rate, n_steps=0.1):
    """
    Applies pitch shifting to the audio.
    
    Parameters:
    audio_data (np.ndarray): The input audio signal.
    sample_rate (int): The sampling rate of the audio signal.
    n_steps (float, optional): The number of pitch shift steps. (default: 0.1)

    Returns:
    pitch_shifted_audio (np.ndarray): The audio signal after pitch shifting.

    """
    pitch_shifted_audio = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)
    return pitch_shifted_audio