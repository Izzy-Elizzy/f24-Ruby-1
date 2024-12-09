import os
import numpy as np
import librosa
import soundfile as sf
import torch
from typing import List, Callable, Dict, Any
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer

class AudioManipulationPipeline:
    def __init__(self, input_folder: str, output_folder: str, wav2vec2_model_name="facebook/wav2vec2-base-960h", embedding_model_name='all-mpnet-base-v2'):
        """
        Initialize the audio manipulation pipeline.

        Args:
            input_folder: Folder containing input audio files.
            output_folder: Folder to save processed audio files.
            wav2vec2_model_name: Name of the pre-trained Wav2Vec2 model to use for embeddings.
            embedding_model_name: Name of the sentence transformer model for short audio comparisons
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
        self.model = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)


        os.makedirs(os.path.join(output_folder, 'original'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'manipulated'), exist_ok=True)

        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder {input_folder} does not exist")

    def _extract_embeddings(self, audio_data: np.ndarray, sample_rate: int, chunk_size=5 * 16000) -> np.ndarray:
        """Extract embeddings, resampling and chunking."""

        target_sr = 16000  # Define the target sample rate
        if sample_rate != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr # Update the sample rate

        if len(audio_data) > chunk_size:
            embeddings = []
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                inputs = self.processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True) # Use resampled rate
                with torch.no_grad():
                    outputs = self.model(**inputs)
                chunk_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(chunk_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)
            embeddings = np.mean(embeddings, axis=0, keepdims=True) # Average chunk embeddings

        else:
            inputs = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True) # Use resampled rate
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        return embeddings

    def _extract_short_embeddings(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract embeddings from short audio data.  This uses a pre-trained model
        that works best on relatively short audio segments (a few seconds).
        For longer files you should segment the audio first.
        """
        # Convert to mono if necessary
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data)

        # Resample if necessary (adjust as needed) - some models have a fixed input sample rate.
        # if sample_rate != 16000:
        #     audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

        # Extract embeddings (assuming your model expects raw audio).
        # Some models might require pre-processing like MFCCs. Adapt as needed.
        embeddings = self.embedding_model.encode(audio_data)  # <--- corrected here
        return embeddings


    def load_audio_files(self) -> List[Dict[str, Any]]:
        """Load audio files from the input folder."""
        audio_files = []
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                full_path = os.path.join(self.input_folder, filename)
                audio_data, sample_rate = librosa.load(full_path, sr=None)
                audio_files.append({
                    'filename': filename,
                    'path': full_path,
                    'data': audio_data,
                    'sample_rate': sample_rate
                })
        return audio_files

    def apply_manipulation(self, audio_files, manipulation_func):
        """Apply audio manipulation to the loaded files."""
        manipulated_files = []
        for audio_file in audio_files:
            manipulated_data = manipulation_func(audio_file['data'], audio_file['sample_rate'])
            base_filename, ext = os.path.splitext(audio_file['filename'])
            manipulated_filename = f"{base_filename}_manipulated{ext}"
            manipulated_path = os.path.join(self.output_folder, 'manipulated', manipulated_filename)
            sf.write(manipulated_path, manipulated_data, audio_file['sample_rate'])
            manipulated_files.append({
                **audio_file,
                'manipulated_data': manipulated_data,
                'manipulated_path': manipulated_path
            })
        return manipulated_files

    def compare_audio_coherence(self, original_files, manipulated_files):
        """Compare audio coherence using cosine similarity of Wav2Vec2 embeddings."""
        coherence_results = []
        for orig, manip in zip(original_files, manipulated_files):
            assert orig['filename'] == manip['filename'], "Mismatched files in comparison"

            orig_embeddings = self._extract_embeddings(orig['data'], orig['sample_rate'])
            manip_embeddings = self._extract_embeddings(manip['manipulated_data'], manip['sample_rate'])

            # Correctly calculate cosine similarity for potentially chunked embeddings
            similarity = np.dot(orig_embeddings[0], manip_embeddings[0]) / (np.linalg.norm(orig_embeddings[0]) * np.linalg.norm(manip_embeddings[0]))

            coherence_results.append({'filename': orig['filename'], 'coherence_score': similarity.item()}) #.item() added here

        return coherence_results

    def run_pipeline(self, manipulation_func: Callable[[np.ndarray, int], np.ndarray]):
        """Run the full audio manipulation pipeline, saving originals."""
        original_files = self.load_audio_files()

        # Save original files to the output folder
        for original_file in original_files:
            original_path = os.path.join(self.output_folder, 'original', original_file['filename'])
            sf.write(original_path, original_file['data'], original_file['sample_rate'])

        manipulated_files = self.apply_manipulation(original_files, manipulation_func)
        coherence_results = self.compare_audio_coherence(original_files, manipulated_files)
        return coherence_results


def your_audio_manipulation(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    REPLACE THIS FUNCTION WITH YOUR SPECIFIC AUDIO MANIPULATION TECHNIQUE

    :param audio_data: Input audio numpy array
    :param sample_rate: Sample rate of the audio
    :return: Manipulated audio numpy array
    """
    # Example: Simple white noise addition
    noise = np.random.normal(0, 0.005, audio_data.shape)
    return audio_data + noise

def main():
    input_folder = './voice_cloner/audio_samples'  # Replace with your input folder
    output_folder = './voice_cloner/output' # Replace with your output folder

    pipeline = AudioManipulationPipeline(input_folder, output_folder)
    results = pipeline.run_pipeline(your_audio_manipulation)

    for result in results:
        print(f"File: {result['filename']}")
        print(f"Coherence Score: {result['coherence_score']:.4f}")
        print("---")


if __name__ == '__main__':
    main()