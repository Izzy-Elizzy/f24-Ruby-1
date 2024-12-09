# equalization.py
from scipy.signal import butter, sosfilt
import numpy as np
 
def apply_equalization(audio_data, sample_rate, freq_center=1000, gain=5, q=0.7):
    """
    Applies equalization to audio file.
    
    Parameters:
    audio_data (np.ndarray): The input audio signal.
    sample_rate (int): The sampling rate of the audio signal.
    freq_center (int): The center frequency of the bandpass filter. (default: 1000 Hz)
    gain (int): Gain applied to the filtered signal. (default: 5 dB)
    q (float): Q-factor of the filter, determining the bandwidth. (default: 0.7)

    Returns:
    equalized_audio (np.ndarray): The equalized audio signal.
    """
    nyquist = 0.5 * sample_rate
    freq_center_norm = freq_center / nyquist
    bandwidth = freq_center_norm / q
    sos = butter(2, [freq_center_norm - bandwidth/2, freq_center_norm + bandwidth/2], 'bandpass', output='sos')
    equalized_audio = sosfilt(sos, audio_data) * (10**(gain/20))
    return equalized_audio
