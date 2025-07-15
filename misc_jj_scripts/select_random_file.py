import random
import glob
import shutil
import os

folder = "E:/joseph/model_3/positives/tp_valid_test"


###todo USE this code for splitting up tps and fps into model 3 and 4


file_list = []
for file in os.listdir(folder):
    # join_path = os.path.join(folder, file)
    file_list.append(file)
# print(file_list)
rand_list = []
i = 0
for file in file_list:
    if i == 75:
        break
    # file_list.append(file)
    random_number = random.randint(0, len(file_list) - 1)

    random_file = file_list[random_number]
    rand_list.append(file_list[random_number])
    file_list.remove(file_list[random_number])
    i += 1
# print(len(rand_list))

#this code moves the files in rand_list to a new directory
for file in rand_list:
    original = "E:/joseph/model_3/positives/tp_valid_test/" + file
    # print(file)
    # print(original)
    # print(len(original))
    filename = file[31:]
    # print(filename)
    destination = "E:/joseph/model_3/positives/TP_VALID_NEW/" + file
    print(destination)
    # print(len(destination))
    shutil.move(original, destination)


#todo

#FOR CREATING TXT FOR TP for "train.txt"

os.chdir("E:/joseph/eventual_test_recordings_or_spare_wgp_recordings/model_3_1.0")
folder = "E:/joseph/eventual_test_recordings_or_spare_wgp_recordings/model_3_1.0/**/*.wav"

i=0
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    print(file)
    filename=file[114:]
    print(filename)
    destination = "E:/joseph/model_3/inference/snippets_for_raven_extra/14/" + filename
    shutil.copy(file, destination)
    i += 1
