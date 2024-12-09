# noise_reduction.py 
import librosa
import numpy as np
 
def apply_noise_reduction(audio_data, sample_rate, n_fft=2048, hop_length=512):
    """
    Applies noise reduction to the audio.
    
    Parameters:
    audio_data (np.ndarray): The input audio signal.
    sample_rate (int): The sampling rate of the audio signal.
    n_fft (int, optional): The number of FFT components for the short-time Fourier transform. (default: 2048) 
    hop_length (int): The hop length for the STFT. (default: 512)

    Returns:
    reduced_noise_audio (np.ndarray): The audio signal after noise reduction.
    
    """
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.mean(np.abs(stft[:, :10]), axis=1, keepdims=True)
    mask = np.maximum(1 - (noise_mag / np.abs(stft)), 0)
    masked_stft = stft * mask
    reduced_noise_audio = librosa.istft(masked_stft, hop_length=hop_length)
    return reduced_noise_audio