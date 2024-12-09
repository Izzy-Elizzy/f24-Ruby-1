"""
Audio Processing Pipeline
========================

This module implements a pipeline for batch processing audio files with customizable
transformations. It follows a three-stage architecture: input, processing, and output.

The script is designed to handle large numbers of audio files while maintaining a clear
separation of concerns between loading, transforming, and saving audio data.

Dependencies:
    - librosa: For audio processing and analysis
    - soundfile: For reading and writing audio files
    - os: For file system operations

Author: Team Ruby
Date: 2024-10-07
"""

import os
import logging
from typing import Tuple
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

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
    """
    Handles all file system operations related to audio processing.
    
    Attributes:
        input_dir (Path): Base directory containing input audio files
        output_dir (Path): Directory where processed files will be saved
        input_format (str): Expected format of input files (e.g., 'flac')
        output_format (str): Desired format for output files (e.g., 'wav')
    """
    
    def __init__(
        self, 
        input_dir: str, 
        output_dir: str, 
        input_format: str = 'flac',
        output_format: str = 'wav'
    ):
        """
        Initialize the AudioFileHandler with input and output directories.
        
        Args:
            input_dir: Path to the input directory
            output_dir: Path to the output directory
            input_format: Extension of input files (default: 'flac')
            output_format: Extension for output files (default: 'wav')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_format = input_format
        self.output_format = output_format
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_input_files(self) -> list[Path]:
        """
        Recursively find all audio files in the input directory.
        
        Returns:
            List of Path objects for all matching audio files
        """
        return list(self.input_dir.rglob(f"*.{self.input_format}"))
    
    def get_output_path(self, input_path: Path) -> Path:
        """
        Generate the output path for a processed file.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Path object for the output file location
        """
        relative_path = input_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path
        output_path = output_path.with_suffix(f'.{self.output_format}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

class AudioLoader:
    """Handles loading audio files and associated operations."""
    
    @staticmethod
    def load_audio(file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file using librosa.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioProcessingError: If file cannot be loaded
        """
        try:
            logger.info(f"Loading audio file: {file_path}")
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            raise AudioProcessingError(f"Failed to load {file_path}: {str(e)}")

class AudioProcessor:
    """Implements audio processing transformations."""
    
    @staticmethod
    def process_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        pitch_steps: float = 0.1,  # Subtle pitch adjustment for realism
        noise_level: float = 0.0005,  # Small amount of noise
        stretch_rate: float = 1.01  # Slight time stretching
    ) -> np.ndarray:
        """
        Apply audio processing transformations.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of the audio
            pitch_steps: Number of steps for pitch shifting (default: 0.3)
            noise_level: Amplitude of added noise (default: 0.001)
            stretch_rate: Rate for time-stretching (default: 1.02)
            
        Returns:
            Processed audio data as NumPy array
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.info("Applying audio transformations")
            
            # Apply pitch shifting
            processed_audio = librosa.effects.pitch_shift(
                audio_data,
                sr=sample_rate,
                n_steps=pitch_steps
            )
            
            # Apply time stretching
            processed_audio = librosa.effects.time_stretch(processed_audio, rate=stretch_rate)
            
            # Add subtle noise
            noise = np.random.normal(0, noise_level, len(processed_audio))
            processed_audio += noise

            return processed_audio
        
        except Exception as e:
            raise AudioProcessingError(f"Processing failed: {str(e)}")

class AudioSaver:
    """Handles saving processed audio files."""
    
    @staticmethod
    def save_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        output_path: Path
    ) -> None:
        """
        Save processed audio to file.
        
        Args:
            audio_data: Processed audio samples
            sample_rate: Sample rate of the audio
            output_path: Where to save the file
            
        Raises:
            AudioProcessingError: If saving fails
        """
        try:
            logger.info(f"Saving processed audio to: {output_path}")
            sf.write(output_path, audio_data, sample_rate)
        except Exception as e:
            raise AudioProcessingError(f"Failed to save {output_path}: {str(e)}")
        
class AudioVisualizer:
    """Class to visualize and save audio waveforms and spectrograms."""
    @staticmethod
    def save_waveform(audio_data: np.ndarray, sample_rate: int, output_path: Path, title: str = "Waveform"):
        """
        Save the waveformof the audio data as a plot.

        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of the audio
            output_path: Path to save the plot
            title: Title of the plot
        """
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
        """
        Save the spectrogram of the audio data as a plot.

        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of the audio
            output_path: Path to save the plot
            title: Title of the plot
        """
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.savefig(output_path)
        plt.close()


def main():
    """Main execution function that ties together the processing pipeline."""
    
    # Initialize components
    file_handler = AudioFileHandler(
        input_dir='./dev-clean/',
        output_dir='./processed_audio/'
    )

    #Create dirctories for storing plots
    plot_dir = Path("./audio_plots")
    original_plot_dir = plot_dir / "original"
    processed_plot_dir = plot_dir / "processed"
    original_plot_dir.mkdir(parents=True, exist_ok=True)
    processed_plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for input_path in file_handler.get_input_files():
        try:
            # Input
            audio_data, sample_rate = AudioLoader.load_audio(input_path)

            # Save original plots
            original_waveform_path = original_plot_dir / f"{input_path.stem}_waveform.png"
            original_spectrogram_path = original_plot_dir / f"{input_path.stem}_spectrogram.png"
            AudioVisualizer.save_waveform(audio_data, sample_rate, original_waveform_path, title=f"Original Waveform: {input_path.name}")
            AudioVisualizer.save_spectrogram(audio_data, sample_rate, original_spectrogram_path, title=f"Original Spectrogram: {input_path.name}")
            
            
            # Process
            processed_audio = AudioProcessor.process_audio(
                audio_data,
                sample_rate,
                pitch_steps=0.1,
                noise_level=0.0005,
                stretch_rate=1.01
            )

            # Save processed plots
            processed_waveform_path = processed_plot_dir / f"{input_path.stem}_waveform.png"
            processed_spectrogram_path = processed_plot_dir / f"{input_path.stem}_spectrogram.png"
            AudioVisualizer.save_waveform(processed_audio, sample_rate, processed_waveform_path, title=f"Processed Waveform:{input_path.name}")
            AudioVisualizer.save_spectrogram(processed_audio, sample_rate, processed_spectrogram_path, title=f"Processed Spectrogram: {input_path.name}")


            
            # Output
            output_path = file_handler.get_output_path(input_path)
            AudioSaver.save_audio(processed_audio, sample_rate, output_path)
            
            logger.info(f"Successfully processed: {input_path}")
            
        except AudioProcessingError as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with {input_path}: {str(e)}")

if __name__ == "__main__":
    main()

    # providing a training voice cloning model (Darell S.)

    pip install TTS # installing Coquoi TTS

    ffmpeg -i input.mp3 -ac 1 -ar 22050 output.wav # could be subject to change
    from TTS.tts.trainer import trainer
    # configurations defined
    config_path = "config_path.json" # path could be changed/modified
    trainer = Trainer(config_path)
    #begin training
    trainer.fit()

    #evaluation
    from TTS.tts.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(model_path= "trained_model_path.pth")
    synthesizer.tts_to_file("Example", file_path= "output_file_path.wav") # subject to change