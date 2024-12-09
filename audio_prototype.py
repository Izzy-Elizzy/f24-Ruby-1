import os
import logging
from typing import Tuple
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass

class AudioFileHandler:
    def __init__(self, input_dir: str, output_dir: str, input_format: str = 'flac', output_format: str = 'wav'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_format = input_format
        self.output_format = output_format
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_input_files(self) -> list[Path]:
        return list(self.input_dir.rglob(f"*.{self.input_format}"))
    
    def get_output_path(self, input_path: Path) -> Path:
        relative_path = input_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path
        output_path = output_path.with_suffix(f'.{self.output_format}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

class AudioLoader:
    @staticmethod
    def load_audio(file_path: Path) -> Tuple[np.ndarray, int]:
        try:
            logger.info(f"Loading audio file: {file_path}")
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            raise AudioProcessingError(f"Failed to load {file_path}: {str(e)}")

class AudioProcessor:
    @staticmethod
    def process_audio(audio_data: np.ndarray, sample_rate: int, pitch_steps: float, noise_level: float, stretch_rate: float) -> np.ndarray:
        """
        Processes audio using pitch shifting, time stretching, and noise addition.

        Args:
            audio_data (np.ndarray): Audio waveform data.
            sample_rate (int): Sample rate of the audio.
            pitch_steps (float): Pitch shift in semitones.
            noise_level (float): Noise amplitude to add.
            stretch_rate (float): Rate for time stretching.

        Returns:
            np.ndarray: Processed audio waveform.
        """
        try:
            logger.info("Applying audio transformations")
            # Pitch shifting
            processed_audio = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_steps)
            # Time stretching
            processed_audio = librosa.effects.time_stretch(processed_audio, rate=stretch_rate)
            # Noise addition
            noise = np.random.normal(0, noise_level, len(processed_audio))
            processed_audio += noise
            return processed_audio
        except Exception as e:
            raise AudioProcessingError(f"Processing failed: {str(e)}")

class AudioSaver:
    @staticmethod
    def save_audio(audio_data: np.ndarray, sample_rate: int, output_path: Path) -> None:
        try:
            logger.info(f"Saving processed audio to: {output_path}")
            sf.write(output_path, audio_data, sample_rate)
        except Exception as e:
            raise AudioProcessingError(f"Failed to save {output_path}: {str(e)}")

class AudioVisualizer:
    @staticmethod
    def save_waveform(audio_data: np.ndarray, sample_rate: int, output_path: Path, title: str = "Waveform"):
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def save_spectrogram(audio_data: np.ndarray, sample_rate: int, output_path: Path, title: str = "Spectrogram"):
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.savefig(output_path)
        plt.close()

def main():
    # Input and output directories
    file_handler = AudioFileHandler(
        input_dir='./dev-clean/',
        output_dir='./processed_audio/'
    )
    plot_dir = Path("./audio_plots")
    original_plot_dir = plot_dir / "original"
    processed_plot_dir = plot_dir / "processed"
    original_plot_dir.mkdir(parents=True, exist_ok=True)
    processed_plot_dir.mkdir(parents=True, exist_ok=True)

    # Parameters for audio transformations
    pitch_steps = 0.5  # Increase or decrease to modify pitch shift
    noise_level = 0.001  # Increase or decrease noise addition
    stretch_rate = 1.02  # Increase or decrease for time stretching

    for input_path in file_handler.get_input_files():
        try:
            # Load audio
            audio_data, sample_rate = AudioLoader.load_audio(input_path)

            # Save original plots
            original_waveform_path = original_plot_dir / f"{input_path.stem}_waveform.png"
            original_spectrogram_path = original_plot_dir / f"{input_path.stem}_spectrogram.png"
            AudioVisualizer.save_waveform(audio_data, sample_rate, original_waveform_path, title=f"Original Waveform: {input_path.name}")
            AudioVisualizer.save_spectrogram(audio_data, sample_rate, original_spectrogram_path, title=f"Original Spectrogram: {input_path.name}")

            # Process audio
            processed_audio = AudioProcessor.process_audio(audio_data, sample_rate, pitch_steps, noise_level, stretch_rate)

            # Save processed plots
            processed_waveform_path = processed_plot_dir / f"{input_path.stem}_waveform.png"
            processed_spectrogram_path = processed_plot_dir / f"{input_path.stem}_spectrogram.png"
            AudioVisualizer.save_waveform(processed_audio, sample_rate, processed_waveform_path, title=f"Processed Waveform: {input_path.name}")
            AudioVisualizer.save_spectrogram(processed_audio, sample_rate, processed_spectrogram_path, title=f"Processed Spectrogram: {input_path.name}")

            # Save processed audio
            output_path = file_handler.get_output_path(input_path)
            AudioSaver.save_audio(processed_audio, sample_rate, output_path)

            logger.info(f"Successfully processed: {input_path}")
        except AudioProcessingError as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with {input_path}: {str(e)}")

if __name__ == "__main__":
    main()