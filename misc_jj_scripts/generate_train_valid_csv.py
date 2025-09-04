import os, csv, shutil, glob
import pandas as pd

### CODE FOR CREATING CSVS IN MODEL DIRECTORIES FOR DATALOADER

os.chdir("D:/PhD/WGP_model/model_dummy/train/")
wrdir = "D:/PhD/WGP_model/model_dummy/train/"
fp = wrdir + "fp/**/*.txt"
fp_valid = wrdir + "fp_valid/**/*.txt"
tp = wrdir + "tp/**/*.txt"
tp_valid = wrdir + "tp_valid/**/*.txt"

#### csv file for training ("train.csv.")

# false positives csv
with open(wrdir + "train_fp.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(["path", "label"])

    for file in glob.glob(fp, recursive=True):
        join_path = os.path.join(fp, file)
        # print(file)
        csvwriter.writerow([file, "0"])

# true positives csv
with open(wrdir + "train_tp.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(["path", "label"])

    for file in glob.glob(tp, recursive=True):
        join_path = os.path.join(tp, file)
        # print(file)
        csvwriter.writerow([file, "1"])

# concatonate both csvs

csv_files = ["train_fp.csv", "train_tp.csv"]

# Create a list to hold the dataframes
df_list = []

for csvs in csv_files:
    file_path = os.path.join(wrdir, csvs)
    try:
        # Try reading the file using default UTF-8 encoding
        df = pd.read_csv(file_path, header=0)
        print(df)
        #df = df.drop(index=1)
        df_list.append(df)
        print(df_list)
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {csvs} because of error: {e}")
    except Exception as e:
        print(f"Could not read file {csvs} because of error: {e}")

# Concatenate all data into one DataFrame
big_df = pd.concat(df_list, ignore_index=True)

# Save the final result to a new CSV file
big_df.to_csv(os.path.join(wrdir, 'training.csv'), index=False)

#delete original csvs
os.remove("train_fp.csv")
os.remove("train_tp.csv")

### csv file for valid ("valid.csv")

# false positives csv
with open(wrdir + "valid_fp.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(["path", "label"])

    for file in glob.glob(fp_valid, recursive=True):
        join_path = os.path.join(fp_valid, file)
        # print(file)
        csvwriter.writerow([file, "0"])

# true positives csv
with open(wrdir + "valid_tp.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(["path", "label"])

    for file in glob.glob(tp_valid, recursive=True):
        join_path = os.path.join(tp_valid, file)
        # print(file)
        csvwriter.writerow([file, "1"])

# concatonate both csvs

csv_files = ["valid_fp.csv", "valid_tp.csv"]

# Create a list to hold the dataframes
df_list = []

for csv in csv_files:
    file_path = os.path.join(wrdir, csv)
    try:
        # Try reading the file using default UTF-8 encoding
        df = pd.read_csv(file_path, header=0)
        print(df)
        #df = df.drop(index=1)
        df_list.append(df)
        print(df_list)
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
big_df.to_csv(os.path.join(wrdir, 'valid.csv'), index=False)

#delete original csvs
os.remove("valid_fp.csv")
os.remove("valid_tp.csv")