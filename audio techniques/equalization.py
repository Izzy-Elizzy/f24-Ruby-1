# equalization.py
from scipy.signal import butter, sosfilt
import numpy as np
 
def apply_equalization(audio_data, sample_rate, freq_center=1000, gain=5, q=0.7):
    """Applies equalization to audio using a bandpass filter."""
    nyquist = 0.5 * sample_rate
    freq_center_norm = freq_center / nyquist
    bandwidth = freq_center_norm / q
    sos = butter(2, [freq_center_norm - bandwidth/2, freq_center_norm + bandwidth/2], 'bandpass', output='sos')
    equalized_audio = sosfilt(sos, audio_data) * (10**(gain/20))
    return equalized_audio
