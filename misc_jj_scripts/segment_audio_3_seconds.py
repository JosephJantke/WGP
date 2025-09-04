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

def cut_audio(input_file, output_file, start_time, end_time):
    """
    Cut parts of an audio file and save it to a new file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the output audio file.
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Cut the audio
    cut_audio = audio[start_time:end_time]

    # Save the cut audio to a new file
    cut_audio.export(output_file, format="wav")

### processing calls in specified directory
base_dir = "D:/PhD/WGP_model/toshiba_dbca_examples_tps/Jims_ARU_good_calls_recordings/screened_recordings"
pattern = os.path.join(base_dir, "**/*.wav")
output_dir = "D:/PhD/WGP_model/toshiba_dbca_examples_tps/Jims_ARU_good_calls_recordings/all_3_second_snippets"
os.makedirs(output_dir, exist_ok=True)

### replace whitespaces with underscore if necessary
for file in glob.glob(pattern, recursive=True):
    destination = file.replace(' ', '_')
    if file != destination:
        shutil.copy(file, destination)
        os.remove(file)

# Segment into 3-second snippets
segment_length_ms = 3000  # 3 seconds

for file in glob.glob(pattern, recursive=True):
    audio_length = print_audio_length(file)

    # Skip files shorter than 3 sec
    if audio_length < 3:
        continue

    # Number of whole 3-sec segments
    num_segments = math.floor(audio_length / 3)

    for i in range(num_segments):
        start_ms = i * segment_length_ms
        end_ms = start_ms + segment_length_ms

        # Build output filename
        base_name = os.path.splitext(os.path.basename(file))[0]
        out_name = f"{base_name}__{i*3}_{(i+1)*3}_seconds.wav"
        output_file = os.path.join(output_dir, out_name)

        cut_audio(file, output_file, start_ms, end_ms)


