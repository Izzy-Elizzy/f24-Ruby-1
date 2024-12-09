Team Ruby - Fall 2024

# Members
- Izzy-Elizzy - Iizalaarab Elhaimeur - ielha003@odu.edu
- jdowe004 - Joshua Dowell - jdowe004@odu.edu
- csDarellS - Darell Styles - dstyl002@odu.edu
- DurelHairston - Durel Hairston - dhair017@odu.edu
- MerlynCode - Dima Bochkarev -  dboch001@odu.edu 

# Requirements
-Python 3.7 or higher
VocalShield uses f-string syntax (f"..."), which was introduced in Python 3.6. Make sure to use Python 3.7 or higher for compatibility.

# Sample Execution & Output
To execute the main functionality of VocalShield, run audio_prototype.py with the necessary input parameters.

If run without command-line arguments, like this:

```
./audio_prototype.py
```

the following usage message will be displayed:

```
Usage: audio_prototype.py --input <path_to_audio_file> --output <path_to_modified_audio> --distortion_level <level>
```

# Example Execution
If run with specified input and output files and a distortion level, such as:

```
./audio_prototype.py --input sample_voice.wav --output protected_voice.wav --distortion_level 5
```

the output will include confirmation messages similar to:

```
Loading input file: sample_voice.wav
Applying distortion level: 5
Saving modified audio to: protected_voice.wav
Process completed successfully.
```




