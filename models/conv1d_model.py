# Import Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from utils.mylogger import logger

# Read train dataframes files
train_df_mfcc1 = pd.read_csv('../csvs/mfccs/MFCC_64_train_meta_16000.csv')
train_df_mfcc2 = pd.read_csv('../csvs/mfccs_augs/mfcc_64_timemask_meta.csv')
train_df_mfcc3 = pd.read_csv('../csvs/mfccs_augs/mfcc_64_freqmask_meta.csv')
train_df_mfcc4 = pd.read_csv('../csvs/mfccs_augs/mfcc_64_timewarp_meta.csv')

# Merge and shuffle dataframes into one csv file
train_df_mfcc = pd.concat([train_df_mfcc1, train_df_mfcc2, train_df_mfcc3, train_df_mfcc4], ignore_index=True)
train_df_mfcc = train_df_mfcc.sample(frac=1).reset_index(drop=True)

# Read a sample to find image dimensions
sample_mfcc = plt.imread(train_df_mfcc['Filepath'][0])
x_size, y_size = np.shape(sample_mfcc)

# Define validation split and batch size
batchsize = 256
val_split = 0.25
datagen = ImageDataGenerator(validation_split=val_split)
# Import of wrapper function to load training dataset
from utils import dataloader, create_model
train_it, val_it, train_size, val_size = dataloader.load_data(datagen, train_df=train_df_mfcc,
                                                              val_split=val_split, batchsize=batchsize,
                                                              mode='training', test_df=None)

# Class weight calculation
my_weights = dataloader.class_weight_calc(train_df_mfcc, class_mode='sklearn')

# Definition of some model parameters
name = 'conv1d_model'
initializer = tf.keras.initializers.he_normal()
reg = tf.keras.regularizers.L2(l2=0.001)
groups = 16

# Model creation
model = create_model.conv1d(x_size, y_size, 1, grouping_num=groups, initializer=initializer, regularizer=reg, name=name,
                            activation='gelu')
model.summary()

# Set callbacks
callback = []
save_best_callback = tf.keras.callbacks.ModelCheckpoint(f'../weights/best_weights.hdf5',
                                                        save_best_only=True, verbose=1)
callback.append(save_best_callback)
early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1)
callback.append(early_stop_callback)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.93,
                                                 min_lr=0, mode='min', cooldown=0, verbose=1)
callback_list = [reduce_lr, callback]

# Hyper-parameter definition
initial_learning_rate = 0.0004
from keras_adabound import AdaBound
opt = AdaBound(learning_rate=initial_learning_rate, final_lr=0.1, amsgrad=True)

loss = 'categorical_crossentropy'
epochs = 75
# Start model training from wrapper function and save trained model
from utils.train_model import training
model, history = training(model, opt, loss, callback_list, my_weights, batchsize, train_size, val_size,
                          epochs, train_it, val_it)
model.save('../models/conv1d_model')

# Evaluate model using test dataset
datagen = ImageDataGenerator()
test_df_mfcc = pd.read_csv('../csvs/mfccs/MFCC_64_test_meta_16000.csv')
test_it, test_size = dataloader.load_data(datagen, train_df=None, test_df=test_df_mfcc, val_split=None,
                                          batchsize=batchsize, mode='test')
score = model.evaluate(test_it, steps=-(-test_size//batchsize), verbose=1)
print("Test loss is:", score[0])
print("Test acc is:", score[1])

# Make predictions using trained model and test set, to obtain necessary metrics
print("Predicting")
y_pred = model.predict(test_it, steps=-(-test_size//batchsize), batch_size=batchsize, verbose=1)
# revert to decimal class representation
y_pred = np.argmax(y_pred, axis=1)

# Metric Calculation
from utils.metrics import confmatrixplot, metric_plot, report_f1score
confmatrixplot(test_df_mfcc, y_pred, Filepath='../metrics_images/conf_run_11th.svg')
metric_plot(history, Filepath='../metrics_images/metrics.svg')
f1score = report_f1score(test_df_mfcc, y_pred)

print("Logging Initiated...")
# Details about current session, to use for logging purposes
min_val_loss = round(early_stop_callback.best, 5)
best_epoch = early_stop_callback.best_epoch
params = model.count_params()
desc = '{}_lr={}_batchsize={}_groups={}' \
       '_usingaugments_epochs{}_reg=l2_' \
       'valsplit={}'.format(opt, initial_learning_rate, batchsize, groups, epochs, val_split)
logger(name, params, round(history.history['accuracy'][best_epoch], 4), round(history.history['loss'][best_epoch], 5),
       round(history.history['val_accuracy'][best_epoch], 4), min_val_loss, round(score[1], 4),
       round(score[0], 4), best_epoch, len(history.history['loss']),
       round(f1score, 4), desc)
print("Logging Complete...")
print("End of file...")
