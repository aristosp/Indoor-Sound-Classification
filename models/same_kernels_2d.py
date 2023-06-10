# Import dependencies
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras_adabound import AdaBound

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# The following code supposes that preprocessing has been done before
# and the spectrograms have been saved in a directory
# Train dataframes
train_df1 = pd.read_csv('../csvs/spectrograms/spec_64_train_meta_16000.csv')
train_df2 = pd.read_csv('../csvs/spec_augs/spec_64_train_timewrap_meta.csv')
train_df3 = pd.read_csv('../csvs/spec_augs/spec_64_train_freqmask_meta.csv')
train_df4 = pd.read_csv('../csvs/spec_augs/spec_64_train_timemask_meta.csv')

# Merge and shuffle sample dataframes
train_df = pd.concat([train_df1, train_df2, train_df3, train_df4], ignore_index=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Test dataframe
test_df = pd.read_csv('../csvs/spectrograms/spec_64_test_meta_16000.csv')
test_df = test_df.sample(frac=1).reset_index(drop=True)

# get x,y dimension size of an image using a random sample to find input shape
sample = plt.imread(train_df['Filepath'][0])
x_size, y_size = np.shape(sample)
# Define training dataset split percentage into training and validation set and batch size
val_split = 0.25
batchsize = 64
datagen = ImageDataGenerator(validation_split=val_split)

# Import wrapper function to load images
from utils import create_model, dataloader
train_it, val_it, train_size, val_size = dataloader.load_data(datagen, train_df=train_df,
                                                              val_split=val_split, batchsize=batchsize,
                                                              mode='training', test_df=None)

# Class weight calculation to deal with dataset class imbalance
my_weights = dataloader.class_weight_calc(train_df, class_mode='sklearn')

# Model Declaration
name = 'symmetrical_kernels'

# Random weight initializer function
initializer = tf.keras.initializers.he_normal()
# Create model using wrapper function
model = create_model.conv2d_same_kernels(name, x_size, y_size, 1, initializer, reg=None, batchsize=batchsize,
                                         logits=False)
print(model.summary())

# Initialize callbacks
callback = []
save_best_callback = tf.keras.callbacks.ModelCheckpoint(f'weights/best_weights_10thrun.hdf5',
                                                        save_best_only=True, verbose=1)
callback.append(save_best_callback)
early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1)
callback.append(early_stop_callback)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.9,
                                                 min_lr=0, mode='min', cooldown=0,
                                                 verbose=1)
callback_list = [callback, reduce_lr]
# Define hyper-parameters
initial_learning_rate = 0.0004
opt = AdaBound(learning_rate=initial_learning_rate, final_lr=0.1, amsgrad=True)
loss = 'categorical_crossentropy'
epochs = 50
# Initialize model training
from utils.train_model import training
model, hist = training(model, opt, loss, callback_list, my_weights, batchsize, train_size, val_size,
                       epochs, train_it, val_it)

# Save model
model.save("models/mymodel2_10thrun")
# Load and evaluate model using test dataset
datagen = ImageDataGenerator()
test_it, test_size = dataloader.load_data(datagen, train_df=None, test_df=test_df, val_split=None,
                                          batchsize=batchsize, mode='test')
score = model.evaluate(test_it, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Make predictions using trained model
print("Predicting")
y_pred = model.predict(test_it, steps=-(-test_size//batchsize), batch_size=batchsize, verbose=1)
# revert to decimal class representation
y_pred = np.argmax(y_pred, axis=1)
# Create necessary plots
from utils.metrics import confmatrixplot, metric_plot, report_f1score
confmatrixplot(test_df, y_pred, Filepath='../2d_convs/conv2d_metrics_samekernel/conf_run10.png')
metric_plot(hist, Filepath='../2d_convs/conv2d_metrics_samekernel/metrics_run10.png')

from utils.mylogger import logger
# Details needed for custom logging function

params = model.count_params()
f1score = report_f1score(test_df, y_pred)
min_val_loss = round(early_stop_callback.best, 5)
best_epoch = early_stop_callback.best_epoch
desc = '{}_lr={}_batchsize={}_reg=none'.format(opt, initial_learning_rate, batchsize)
print("Logging Initiated...")
logger(name, params, round(hist.history['accuracy'][best_epoch], 4),
       round(hist.history['loss'][best_epoch], 5), round(hist.history['val_accuracy'][best_epoch], 4),
       min_val_loss, round(score[1], 4), round(score[0], 4), best_epoch, len(hist.history['loss']),
       round(f1score, 4), desc)
print("Logging Complete...")
print("End of file...")
