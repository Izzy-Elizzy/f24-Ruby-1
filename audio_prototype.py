import os
import librosa
import soundfile as sf

# Define the path to the folder containing your audio files
base_dir = '/Users/dowell/Desktop/VocalShield Prototype/dev-clean/'

# Define the path to the output folder where processed files will be saved
output_dir = '/Users/dowell/Desktop/VocalShield Prototype/processed_audio/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to process each audio file
def process_audio(file_path):
    try:
        # Load the audio file
        print(f"Loading file: {file_path}")  # Debugging line
        y, sr = librosa.load(file_path, sr=None)

        # Apply some distortion, e.g., pitch shifting
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)  # Corrected line

        # Create the output file path in the output folder
        relative_path = os.path.relpath(file_path, base_dir)
        output_file = os.path.join(output_dir, relative_path.replace('.flac', '_processed.wav'))

        # Ensure the subdirectory exists in the output folder
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the processed audio file
        sf.write(output_file, y_shifted, sr)
        print(f"Processed {file_path} and saved as {output_file}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Walk through the folders and process each file
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.flac'):  # Process only .flac files
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")  # Debugging line
            process_audio(file_path)