import tkinter as tk
import soundfile as sf
from tkinter import filedialog, messagebox
import os
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

class AudioProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Processing GUI")
        self.root.geometry("600x400")
        
        # Variables
        self.processed_file = None
        self.cloned_file = None
        
        # File Input
        tk.Label(root, text="Processed File:").pack(pady=5)
        tk.Button(root, text="Browse Processed", command=self.browse_processed).pack(pady=5)
        
        tk.Label(root, text="Cloned File:").pack(pady=5)
        tk.Button(root, text="Browse Cloned", command=self.browse_cloned).pack(pady=5)
        
        # Processing Options
        tk.Label(root, text="Processing Parameters").pack(pady=10)
        tk.Label(root, text="Pitch Steps").pack()
        self.pitch_steps = tk.Entry(root)
        self.pitch_steps.insert(0, "0.1")
        self.pitch_steps.pack()
        
        tk.Label(root, text="Noise Level").pack()
        self.noise_level = tk.Entry(root)
        self.noise_level.insert(0, "0.0005")
        self.noise_level.pack()
        
        tk.Label(root, text="Stretch Rate").pack()
        self.stretch_rate = tk.Entry(root)
        self.stretch_rate.insert(0, "1.01")
        self.stretch_rate.pack()
        
        # Buttons
        tk.Button(root, text="Process Audio", command=self.process_audio).pack(pady=20)
        tk.Button(root, text="Quit", command=root.quit).pack(pady=5)

    def browse_processed(self):
        self.processed_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac")])
        if self.processed_file:
            messagebox.showinfo("Selected", f"Processed File: {self.processed_file}")

    def browse_cloned(self):
        self.cloned_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if self.cloned_file:
            messagebox.showinfo("Selected", f"Cloned File: {self.cloned_file}")

    def process_audio(self):
        if not self.processed_file or not self.cloned_file:
            messagebox.showerror("Error", "Please select both processed and cloned files.")
            return

        try:
            # Load processed audio
            processed_audio, sample_rate = librosa.load(self.processed_file, sr=None)

            # Apply transformations
            pitch_steps = float(self.pitch_steps.get())
            noise_level = float(self.noise_level.get())
            stretch_rate = float(self.stretch_rate.get())

            processed_audio = librosa.effects.pitch_shift(processed_audio, sr=sample_rate, n_steps=pitch_steps)
            processed_audio = librosa.effects.time_stretch(processed_audio, rate=stretch_rate)
            processed_audio += np.random.normal(0, noise_level, len(processed_audio))
            
            # Save processed audio
            output_path = Path("./processed_output.wav")
            sf.write(output_path, processed_audio, sample_rate)
            
            messagebox.showinfo("Success", f"Processed audio saved at {output_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessingGUI(root)
    root.mainloop()
