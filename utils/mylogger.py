import csv
import pandas as pd
import os
def csv_creator(filename, fields, rows):
    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(rows)


def logger(model_name, num_params, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, best_epoch, epochs, macro_f1score , desc):
    fields = ['Model', 'Number Of Params', 'Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test Accuracy', 'Test Loss', 'Best Epoch','Epochs', 'Macro-F1 score', 'Description']
    if 'Logs.csv' not in os.listdir():
        row = [model_name, num_params, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, best_epoch, epochs, macro_f1score , desc]
        csv_creator('Logs.csv', fields, row)
    else:
        row = [model_name, num_params, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, best_epoch, epochs, macro_f1score , desc]
        with open(r'Logs.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)