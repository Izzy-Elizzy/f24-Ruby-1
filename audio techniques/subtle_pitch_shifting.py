import librosa
 
def apply_subtle_pitch_shifting(audio_data, sample_rate, n_steps=0.1):
    """Applies subtle pitch shifting to audio."""
    pitch_shifted_audio = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)
    return pitch_shifted_audio