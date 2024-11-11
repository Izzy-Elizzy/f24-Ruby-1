import os
import subprocess
from pathlib import Path

def convert_m4a_to_wav(input_path, output_path):
    try:
        # Using subprocess to call ffmpeg directly
        command = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            output_path,
            '-y'  # Overwrite output file if it exists
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error converting {input_path}: {e}")
        return False

def convert_and_rename_audio_files(folder_path):
    # Convert folder_path to Path object
    folder = Path(folder_path)
    
    # Get all m4a files
    m4a_files = list(folder.glob('*.m4a'))
    
    print(f'Folder path: {folder}')
    print(f'M4A files found: {len(m4a_files)}')
    
    # Convert all m4a files to wav
    successful_conversions = []
    
    for m4a_file in m4a_files:
        wav_file = m4a_file.with_suffix('.wav')
        print(f'Converting: {m4a_file.name} -> {wav_file.name}')
        
        if convert_m4a_to_wav(str(m4a_file), str(wav_file)):
            successful_conversions.append(wav_file)
            # Delete original m4a file after successful conversion
            m4a_file.unlink()
            print(f'Successfully converted and deleted: {m4a_file.name}')
        else:
            print(f'Failed to convert: {m4a_file.name}')
    
    # Rename all wav files sequentially
    successful_conversions.sort()
    for index, wav_file in enumerate(successful_conversions, start=1):
        new_name = folder / f'{index}.wav'
        wav_file.rename(new_name)
        print(f'Renamed {wav_file.name} to {new_name.name}')

if __name__ == "__main__":
    folder_path = "./voice_cloner/audio_samples"  # Your folder path
    convert_and_rename_audio_files(folder_path)
    print("Audio files converted and renamed successfully!")