from tqdm import tqdm
import re
import os
import PIL
import shutil
import glob
import csv
import pandas as pd

#todo this code enters a 1 or 0 into the validation column of the text file
os.chdir("E:/joseph/model_x/inference/model_2/selection_tables")
home = "E:/joseph/model_x/inference/model_2/selection_tables"
folder = "E:/joseph/model_x/inference/model_2/selection_tables/**/*.txt"
for file in glob.glob(folder, recursive=True):
    # print(file)
    textfilename = file[55:78]
    csvfilename = file[53:78] + '.csv'
    print(textfilename)
    # print(csvfilename)
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    content = content.split("\n")
    # print(len(content))

    audacity_labels = "C:/wgp_project/audacity_labels/" + textfilename + ".txt"
    # print(audacity_labels)
    # for creating list of all annotations
    fp = open(audacity_labels, "r")
    data = fp.read()
    # print(data)
    mylist = data.replace('\n', ' ')
    mylist = re.split('\t| ', mylist)
    mylist = ([s for s in mylist if s != '\\'])
    mylist = ([s for s in mylist if s != ''])
    mylist = [float(x) for x in mylist]
    # make a list of lists for annotations:
    def split(list_a, chunk_size):
        for i in tqdm(range(0, len(list_a), chunk_size), desc='splitting...'):
            yield list_a[i:i + chunk_size]

    chunk_size = 5
    annotations_list_old = list(split(mylist, chunk_size))
    chunk_size = 5
    annotations_list_old = list(split(mylist, chunk_size))
    annotations_list = []
    for substring in annotations_list_old:
        res = substring[: len(substring) - 3]
        # print(str(res))
        annotations_list.append(res)
    with open(csvfilename, 'w', newline='') as out_file:
        csv_out = csv.writer(out_file)
        column_names = ['file', 'prob', 'predicted', 'label']
        writer = csv.DictWriter(out_file, fieldnames=column_names)
        writer.writeheader()
        for line in content:
            # print(line)
            time = line[-14:-7]
            # print(time)
            filename = line[-28:-7]
            # print(filename)
            probability = line[-22:-20]
            # print(probability)
            time = float(time)                                                                                          #todo need to remove empty line at bottom of text files!!!
            range_of_annotations = len(annotations_list)
            randlist = []
            for t0, t1 in annotations_list:
                if t0 - 1.5 <= time <= t1 + 1.5:
                    csv_out.writerow([filename, probability, "1", "1"])
                else:
                    # csv_out.writerow([filename, probability, "1", "0"])
                    randlist.append("1")
                if len(randlist) == range_of_annotations:
                    csv_out.writerow([filename, probability, "1", "0"])
                
# #todo this code creates annotation_list
# audacity_labels = "C:/wgp_project/audacity_labels/IFRP92_20210402_175500.txt"
# # for creating list of all annotations
# fp = open(audacity_labels, "r")
# data = fp.read()
# # print(data)
# mylist = data.replace('\n', ' ')
# mylist = re.split('\t| ', mylist)
# mylist = ([s for s in mylist if s != '\\'])
# mylist = ([s for s in mylist if s != ''])
# mylist = [float(x) for x in mylist]
# # make a list of lists for annotations:
# def split(list_a, chunk_size):
#     for i in tqdm(range(0, len(list_a), chunk_size), desc='splitting...'):
#         yield list_a[i:i + chunk_size]
# chunk_size = 5
# annotations_list_old = list(split(mylist, chunk_size))
# chunk_size = 5
# annotations_list_old = list(split(mylist, chunk_size))
#
# annotations_list = []
# for substring in annotations_list:
#     res = substring[: len(substring) - 3]
#     print(str(res))
#     annotations_list.append(res)

import pandas as pd
import os

#TODO code concatonates all csv files
# replace with your folder's path
folder_path = r'E:\joseph\model_x\inference\model_2\selection_tables'

all_files = os.listdir(folder_path)

# Filter out non-CSV files
csv_files = [f for f in all_files if f.endswith('.csv')]

# Create a list to hold the dataframes
df_list = []

for csv in csv_files:
    file_path = os.path.join(folder_path, csv)
    try:
        # Try reading the file using default UTF-8 encoding
        df = pd.read_csv(file_path)
        df_list.append(df)
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")
    except Exception as e:
        print(f"Could not read file {csv} because of error: {e}")

# Concatenate all data into one DataFrame
big_df = pd.concat(df_list, ignore_index=True)

# Save the final result to a new CSV file
big_df.to_csv(os.path.join(folder_path, 'combined_file.csv'), index=False)