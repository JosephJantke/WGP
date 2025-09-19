from pydub import AudioSegment
import os
import shutil
import glob
import math

### find audio length
def print_audio_length(file_path):
    """
    Loads an audio file and prints its duration in milliseconds and seconds.

    Args:
        file_path (str): The path to the audio file.
    """
    try:
        audio = AudioSegment.from_file(file_path)

        # Get duration in milliseconds
        duration_ms = len(audio)
        #
        # print(f"Audio length in milliseconds: {duration_ms} ms")

        # Get duration in seconds
        duration_seconds = audio.duration_seconds
        return (duration_seconds)
        # print(f"Audio length in seconds: {duration_seconds:.2f} seconds")

    except Exception as e:
        print(f"Error processing audio file: {e}")

### processing calls in specified directory
base_dir = "D:/PhD/WGP_model/toshiba_dbca_examples_tps/CANP_wgp_calls_recordings/calls_by_place/screened_recordings"
folder = os.path.join(base_dir, "**/*.wav")

### replace whitespaces with underscore if necessary
for file in glob.glob(folder, recursive=True):
    audio_length = print_audio_length(file)
    print(audio_length)
    if audio_length > 20:
        os.remove(file)
