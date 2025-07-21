from pydub import AudioSegment
import os
import shutil
import glob

#find audio length
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
        print(f"Audio length in milliseconds: {duration_ms} ms")

        # Get duration in seconds
        duration_seconds = audio.duration_seconds
        print(f"Audio length in seconds: {duration_seconds:.2f} seconds")

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


# audio for trimming:
# recording = 'D:/PhD/WGP_model/from_toshiba_dbca_examples/ET netline 20230419 pm1001 D'
# input_file = recording + '.wav'
# output_file = recording + "_5_seconds" + ".wav"
# start_time = 0
# end_time = 5000
#
# cut_audio(input_file, output_file, start_time, end_time)


#processing calls in directory
os.chdir("D:/PhD/WGP_model/from_toshiba_dbca_examples/CANP_monitoring_ARU_calls")
folder = "D:/PhD/WGP_model/from_toshiba_dbca_examples/CANP_monitoring_ARU_calls/**/*.wav"

#replace whitespaces with underscore if necessary
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    original = file
    print(original)
    destination = original.replace(' ', '_')
    print(destination)
    shutil.copy(original, destination)
    os.remove(original)

#segment audio into 5 second snippets if necessary
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    # print(file)

    # 0-5 second segment
    input_file = file
    output_file = file[:-4] + "_0_5_seconds" + ".wav"
    # print(output_file)
    start_time = 0
    end_time = 5000
    cut_audio(input_file, output_file, start_time, end_time)

    # todo 5-10 second segment