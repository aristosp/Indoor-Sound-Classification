# Import Dependencies
import tensorflow as tf
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from keras_adabound import AdaBound

# Read training and test Dataframes
train_df1 = pd.read_csv('../csvs/spectrograms/spec_64_train_meta_16000.csv.csv')
train_df2 = pd.read_csv('../csvs/spec_augs/spec_64_train_timewrap_meta.csv')
train_df3 = pd.read_csv('../csvs/spec_augs/spec_64_train_timemask_meta.csv')
train_df4 = pd.read_csv('../csvs/spec_augs/spec_64_train_freqmask_meta.csv')
train_df = pd.concat([train_df1, train_df2, train_df3, train_df4], ignore_index=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)

test_df = pd.read_csv('../csvs/spectrograms/spec_64_test_meta_16000.csv')
test_df = test_df.sample(frac=1).reset_index(drop=True)

# get x,y size of image using a random sample
sample = plt.imread(train_df['Filepath'][0])
x_size, y_size = np.shape(sample)

# Initialize data loader, define batch size and split between training and validation set
batchsize = 256
val_split = 0.25
datagen = ImageDataGenerator(validation_split=val_split)
from utils import create_model, dataloader
print("Load Data")
train_it, val_it, train_size, val_size = dataloader.load_data(datagen, train_df=train_df,
                                                              val_split=val_split, batchsize=batchsize,
                                                              mode='training', test_df=None)

class_weights = dataloader.class_weight_calc(train_df, class_mode='sklearn')

# Model Declaration
name = 'conv2d_diffkernels_10th'
initializer = tf.keras.initializers.he_normal()
model = create_model.conv2d_diff_kernels(name, x_size, y_size, 1, initializer,
                                          reg=None, batchsize=batchsize, logits=False)

# Initialize callbacks
callback = []
save_best_callback = tf.keras.callbacks.ModelCheckpoint(f'../2d_convs/weights/best_weights_diff_kernels.hdf5', save_best_only=True, verbose=1)
callback.append(save_best_callback)
early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1)
callback.append(early_stop_callback)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.92,
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
model, hist = training(model, opt, loss, callback_list, class_weights, batchsize, train_size, val_size,
                       epochs, train_it, val_it)

# Save trained model
model.save("../2d_convs/models/diff_kernels_10th")

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
confmatrixplot(test_df, y_pred, Filepath='../2d_convs/conv2d_metrics_diffkernel/conf_run10.png')
metric_plot(hist, Filepath='../2d_convs/conv2d_metrics_diffkernel/metrics_run10.png')

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
