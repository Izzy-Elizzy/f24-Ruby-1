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

    #noise_reduction.py 
    import librosa
    import numpy as np
 
def apply_noise_reduction(audio_data, sample_rate, n_fft=2048, hop_length=512):
    """Applies noise reduction using spectral subtraction."""
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.mean(np.abs(stft[:, :10]), axis=1, keepdims=True)
    mask = np.maximum(1 - (noise_mag / np.abs(stft)), 0)
    masked_stft = stft * mask
    reduced_noise_audio = librosa.istft(masked_stft, hop_length=hop_length)
    return reduced_noise_audio