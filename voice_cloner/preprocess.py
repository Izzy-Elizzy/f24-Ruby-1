import os
import librosa
import soundfile as sf
from pathlib import Path

def preprocess_audio(input_path, output_path):
    # Convert to Path objects
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all wav files
    wav_files = list(input_dir.glob('*.wav'))
    print(f'Found {len(wav_files)} WAV files to process')
    
    for wav_file in wav_files:
        try:
            print(f'Processing: {wav_file.name}')
            
            # Load the .wav file
            y, sr = librosa.load(str(wav_file), sr=22050)
            
            # Trim silence
            trimmed_audio, _ = librosa.effects.trim(y, top_db=20)
            
            # Normalize audio
            normalized_audio = librosa.util.normalize(trimmed_audio)
            
            # Save processed .wav file to the output folder
            output_file = output_dir / wav_file.name
            sf.write(str(output_file), normalized_audio, sr, subtype='PCM_16')
            
            print(f'Successfully processed: {wav_file.name}')
            
        except Exception as e:
            print(f'Error processing {wav_file.name}: {e}')

if __name__ == "__main__":
    input_path = "./voice_cloner/audio_samples"    # Your input folder with WAV files
    output_path = "./voice_cloner/processed_wav"   # Where to save processed files
    
    preprocess_audio(input_path, output_path)
    print("Audio preprocessing completed!")