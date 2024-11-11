import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
from tqdm import tqdm

def transcribe_wav_files(wav_directory, output_path=None):
    # Convert to Path object
    wav_dir = Path(wav_directory)
    
    # If output_path is not specified, create list.txt in wav_directory
    if output_path is None:
        output_path = wav_dir / "list.txt"
    else:
        output_path = Path(output_path)
    
    # Get list of wav files and count them
    wav_files = list(wav_dir.glob('*.wav'))
    wav_files.sort(key=lambda x: int(x.stem))  # Sort numerically
    
    print(f"Found {len(wav_files)} WAV files to process")
    print("Loading Wav2Vec2 model and processor...")
    
    # Initialize the wav2vec model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    
    # Initialize the list to store file paths and transcripts
    file_and_transcripts = []
    
    # Create a progress bar
    print("Transcribing audio files...")
    for wav_file in tqdm(wav_files, desc="Processing"):
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(str(wav_file))
            waveform = waveform.squeeze()  # Squeeze the batch dimension
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # Process through wav2vec
            input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
            input_values = input_values.to(device)
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = processor.decode(predicted_ids[0])
            
            # Format the path according to your requirement
            # Note: Adjusting the path format to match your requirement
            formatted_path = f"/content/TTS-TT2/wavs/{wav_file.name}|{transcript}"
            file_and_transcripts.append(formatted_path)
            
        except Exception as e:
            print(f"\nError processing {wav_file.name}: {e}")
            continue
    
    # Write the file paths and transcripts to the output file
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            for line in file_and_transcripts:
                f.write(f"{line}\n")
        print(f"\nTranscriptions saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving transcriptions: {e}")
    
    return file_and_transcripts

if __name__ == "__main__":
    wav_directory = "./voice_cloner/audio_samples"    # Your WAV files directory
    output_file = "./voice_cloner/list.txt"          # Where to save the transcript list
    
    transcripts = transcribe_wav_files(wav_directory, output_file)
    print("Transcription completed!")