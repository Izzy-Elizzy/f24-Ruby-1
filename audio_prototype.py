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
    def process_audio(audio_data: np.ndarray, sample_rate: int, pitch_steps: float = 0.1, noise_level: float = 0.0005, stretch_rate: float = 1.01) -> np.ndarray:
        try:
            logger.info("Applying audio transformations")
            processed_audio = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_steps)
            processed_audio = librosa.effects.time_stretch(processed_audio, rate=stretch_rate)
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

def extract_audio_features(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        rms = np.sqrt(np.mean(audio_data ** 2))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        return {"RMS": rms, "Spectral_Centroid": spectral_centroid, "Zero_Crossing_Rate": zero_crossing_rate}
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def main():
    file_handler = AudioFileHandler(
        input_dir='./dev-clean/',
        output_dir='./processed_audio/'
    )
    plot_dir = Path("./audio_plots")
    original_plot_dir = plot_dir / "original"
    processed_plot_dir = plot_dir / "processed"
    original_plot_dir.mkdir(parents=True, exist_ok=True)
    processed_plot_dir.mkdir(parents=True, exist_ok=True)

    comparison_dir = Path("./comparison_audio")
    processed_audio_dir = comparison_dir / "processed"
    cloned_audio_dir = comparison_dir / "cloned"
    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    cloned_audio_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_results = []
    
    for input_path in file_handler.get_input_files():
        try:
            audio_data, sample_rate = AudioLoader.load_audio(input_path)
            original_waveform_path = original_plot_dir / f"{input_path.stem}_waveform.png"
            original_spectrogram_path = original_plot_dir / f"{input_path.stem}_spectrogram.png"
            AudioVisualizer.save_waveform(audio_data, sample_rate, original_waveform_path, title=f"Original Waveform: {input_path.name}")
            AudioVisualizer.save_spectrogram(audio_data, sample_rate, original_spectrogram_path, title=f"Original Spectrogram: {input_path.name}")

            processed_audio = AudioProcessor.process_audio(audio_data, sample_rate)
            processed_waveform_path = processed_plot_dir / f"{input_path.stem}_waveform.png"
            processed_spectrogram_path = processed_plot_dir / f"{input_path.stem}_spectrogram.png"
            AudioVisualizer.save_waveform(processed_audio, sample_rate, processed_waveform_path, title=f"Processed Waveform:{input_path.name}")
            AudioVisualizer.save_spectrogram(processed_audio, sample_rate, processed_spectrogram_path, title=f"Processed Spectrogram: {input_path.name}")

            output_path = file_handler.get_output_path(input_path)
            AudioSaver.save_audio(processed_audio, sample_rate, output_path)

            processed_comparison_path = processed_audio_dir / output_path.name
            output_path.rename(processed_comparison_path)

            cloned_audio_path = cloned_audio_dir / f"{input_path.stem}_cloned.mp3"
            if cloned_audio_path.exists():
                processed_features = extract_audio_features(processed_comparison_path)
                cloned_features = extract_audio_features(cloned_audio_path)
                if processed_features and cloned_features:
                    comparison_results.append({
                        "File": input_path.name,
                        "Processed_RMS": processed_features["RMS"],
                        "Cloned_RMS": cloned_features["RMS"],
                        "Processed_Spectral_Centroid": processed_features["Spectral_Centroid"],
                        "Cloned_Spectral_Centroid": cloned_features["Spectral_Centroid"],
                        "Processed_Zero_Crossing_Rate": processed_features["Zero_Crossing_Rate"],
                        "Cloned_Zero_Crossing_Rate": cloned_features["Zero_Crossing_Rate"]
                    })
            else:
                logger.warning(f"Cloned file not found for {input_path.name}. Skipping comparison.")
            logger.info(f"Successfully processed: {input_path}")
        except AudioProcessingError as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with {input_path}: {str(e)}")

    if comparison_results:
        output_csv = comparison_dir / "audio_comparison_results.xlsx"
        df = pd.DataFrame(comparison_results)
        df.to_excel(output_csv, index=False)
        logger.info(f"Comparison results saved to {output_csv}")
    else:
        logger.info("No comparison results to save.")

if __name__ == "__main__":
    main()