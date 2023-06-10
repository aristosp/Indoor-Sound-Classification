import os
import numpy as np
import csv
import pandas as pd
import re


# Create csv file of image directory and corresponding label
def sorted_natural(data):
    """
    This function sorts its input naturally instead of alphabetically
    :param data: Input to be sorted
    :return: it's input, sorted naturally
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def csv_creator(filename, fields, rows):
    """
    This function creates a csv file, filling it with the contents of fields and rows
    :param filename: Path and name to save the created file
    :param fields: Data to fill the fields of the csv file
    :param rows: Data to fill the rows of the csv file
    :return: Null
    """
    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(rows)


# flag = 0 denotes use of spectrograms
# flag = 1 denotes use of mfccs
# First for training dataset
flag = 0
if flag == 0:
    train_path = 'E:/Spec/64/Test_16000/'
    # test_path = 'E:/Spec/64/Test/'
    # test_meta = pd.read_csv('../csvs/test_meta.csv')
else:
    train_path = 'E:/MFCC_augments/128/time_warp/'
    test_path = 'E:/MFCC_augments/128/freq_mask/'
    # test_meta = pd.read_csv('../csvs/test_meta.csv')
train_meta = pd.read_csv('../csvs/test_meta.csv')  # read metadata
print(train_meta.head())
char_labels = train_meta['Label']
train_fields = ['Filepath', 'Label']
spec_paths = os.listdir(train_path)
spec_paths = sorted_natural(spec_paths)
train_paths = []
for path in spec_paths:
    train_paths.append(train_path+path)
file_name = 'E:/csvs/test_meta.csv'
train_rows = []
for i in range(len(train_paths)):
    train_rows.append([train_paths[i], char_labels[i]])
# Save csv file
csv_creator(file_name, train_fields, train_rows)

# Then for test dataset
# test_meta = pd.read_csv('../csvs/test_meta.csv')
# char_test_labels = test_meta['Label']
# test_fields = ['Filepath', 'Label']
# spec_test_paths = os.listdir(test_path)
# # spec_test_paths = sorted_natural(spec_test_paths)
# test_paths = []
# test_file_name = 'E:/csvs/test.csv'
# for path in spec_test_paths:
#     test_paths.append(test_path + path)
# test_rows = []
# for i in range(len(test_paths)):
#     test_rows.append([test_paths[i], char_test_labels[i]])
# # Save csv file
# csv_creator(test_file_name, test_fields, test_rows)
# print("End of file")

