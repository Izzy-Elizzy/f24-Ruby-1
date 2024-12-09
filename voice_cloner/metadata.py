import os
import shutil
import taglib
from pathlib import Path

def update_metadata(input_folder, output_folder):
    # Convert to Path objects
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the count of wav files in the input folder
    wav_files = list(input_path.glob('*.wav'))
    file_count = len(wav_files)

    for i in range(1, file_count + 1):
        input_file = input_path / f"{i}.wav"
        output_file = output_path / f"{i}.wav"

        if input_file.exists():
            try:
                # Load WAV file and update metadata
                with taglib.File(str(input_file)) as audio:
                    # Set the title to match the file name without the extension
                    audio.tags["TITLE"] = [f"{i}"]
                    # Set the track number to match the file name without the extension
                    audio.tags["TRACKNUMBER"] = [f"{i}"]
                    # Save updated WAV file
                    audio.save()

                # Copy the updated file to the output folder
                shutil.copy2(str(input_file), str(output_file))
                print(f"Updated metadata for {i}.wav: title='{i}', track number={i}")
            
            except Exception as e:
                print(f"Error processing {i}.wav: {e}")
        else:
            print(f"File {i}.wav not found.")

if __name__ == "__main__":
    input_folder = "./voice_cloner/audio_samples" 
    output_folder = "./voice_cloner/final_wav"    
    update_metadata(input_folder, output_folder)
    print("Metadata update completed!")