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

### processing calls in "CANP_monitoring_ARU_recordings_unsegmented" directory
os.chdir("D:/PhD/WGP_model/from_toshiba_dbca_examples/CANP_monitoring_ARU_recordings_unsegmented")
folder = "D:/PhD/WGP_model/from_toshiba_dbca_examples/CANP_monitoring_ARU_recordings_unsegmented/**/*.wav"

#replace whitespaces with underscore if necessary
# for file in glob.glob(folder, recursive=True):
#     join_path = os.path.join(folder, file)
#     original = file
#     print(original)
#     destination = original.replace(' ', '_')
#     print(destination)
#     shutil.copy(original, destination)
#     os.remove(original)

###segment audio into 5 second snippets if necessary NOTE: DELETES THE ORIGINAL AUDIO FILE
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    # print(file)
    audio_length = print_audio_length(file)

    #remove audio if less than five seconds
    if audio_length < 5:
        continue

    # 0-5 second segment
    if audio_length > 5:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__0_5_seconds" + ".wav"
        # print(output_file)
        start_time = 0
        end_time = 5000
        cut_audio(input_file, output_file, start_time, end_time)

    # 5-10 second segment
    if audio_length > 10:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__5_10_seconds" + ".wav"
        # print(output_file)
        start_time = 5000
        end_time = 10000
        cut_audio(input_file, output_file, start_time, end_time)

    # 10-15 second segment
    if audio_length > 15:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__10_15_seconds" + ".wav"
        # print(output_file)
        start_time = 10000
        end_time = 15000
        cut_audio(input_file, output_file, start_time, end_time)

    # 15-20 second segment
    if audio_length > 20:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__15_20_seconds" + ".wav"
        # print(output_file)
        start_time = 15000
        end_time = 20000
        cut_audio(input_file, output_file, start_time, end_time)

    # 20-25 second segment
    if audio_length > 25:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__20_25_seconds" + ".wav"
        # print(output_file)
        start_time = 20000
        end_time = 25000
        cut_audio(input_file, output_file, start_time, end_time)

    # 25-30 second segment
    if audio_length > 30:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__25_30_seconds" + ".wav"
        # print(output_file)
        start_time = 25000
        end_time = 30000
        cut_audio(input_file, output_file, start_time, end_time)

    # 30-35 second segment
    if audio_length > 35:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__30_35_seconds" + ".wav"
        print(output_file)
        start_time = 30000
        end_time = 35000
        cut_audio(input_file, output_file, start_time, end_time)

    # 35-40 second segment
    if audio_length > 40:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__35_40_seconds" + ".wav"
        # print(output_file)
        start_time = 35000
        end_time = 40000
        cut_audio(input_file, output_file, start_time, end_time)

    # 40-45 second segment
    if audio_length > 45:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__40_45_seconds" + ".wav"
        # print(output_file)
        start_time = 40000
        end_time = 45000
        cut_audio(input_file, output_file, start_time, end_time)

    # 45-50 second segment
    if audio_length > 50:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__45_50_seconds" + ".wav"
        # print(output_file)
        start_time = 45000
        end_time = 50000
        cut_audio(input_file, output_file, start_time, end_time)

    # 50-55 second segment
    if audio_length > 55:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__50_55_seconds" + ".wav"
        # print(output_file)
        start_time = 50000
        end_time = 55000
        cut_audio(input_file, output_file, start_time, end_time)

    # 55-60 second segment
    if audio_length > 60:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__55_60_seconds" + ".wav"
        # print(output_file)
        start_time = 55000
        end_time = 60000
        cut_audio(input_file, output_file, start_time, end_time)

    # 60-65 second segment
    if audio_length > 65:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__60_65_seconds" + ".wav"
        # print(output_file)
        start_time = 60000
        end_time = 65000
        cut_audio(input_file, output_file, start_time, end_time)

    # 65-70 second segment
    if audio_length > 70:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__65_70_seconds" + ".wav"
        # print(output_file)
        start_time = 65000
        end_time = 70000
        cut_audio(input_file, output_file, start_time, end_time)

    # 70-75 second segment
    if audio_length > 75:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__70_75_seconds" + ".wav"
        # print(output_file)
        start_time = 70000
        end_time = 75000
        cut_audio(input_file, output_file, start_time, end_time)

    # 75-80 second segment
    if audio_length > 80:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__75_80_seconds" + ".wav"
        # print(output_file)
        start_time = 75000
        end_time = 80000
        cut_audio(input_file, output_file, start_time, end_time)

    # 80-85 second segment
    if audio_length > 85:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__80_85_seconds" + ".wav"
        # print(output_file)
        start_time = 80000
        end_time = 85000
        cut_audio(input_file, output_file, start_time, end_time)

    # 85-90 second segment
    if audio_length > 90:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__85_90_seconds" + ".wav"
        # print(output_file)
        start_time = 85000
        end_time = 90000
        cut_audio(input_file, output_file, start_time, end_time)

    # 90-95 second segment
    if audio_length > 95:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__90_95_seconds" + ".wav"
        # print(output_file)
        start_time = 90000
        end_time = 95000
        cut_audio(input_file, output_file, start_time, end_time)

    # 95-100 second segment
    if audio_length > 100:
        input_file = file
        output_file = file[:44] + "CANP_monitoring_ARU_recordings_segmented/" + file[70:-4] + "__95_100_seconds" + ".wav"
        # print(output_file)
        start_time = 95000
        end_time = 100000
        cut_audio(input_file, output_file, start_time, end_time)

    #remove the original whole file
    os.remove(file)

###