import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# Bug in tf 2.8 the following errors
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# Read Dataframes
train_df1 = pd.read_csv('../csvs/spectrograms/spec_64_train_meta_16000.csv')
train_df2 = pd.read_csv('../csvs/spec_augs/spec_64_train_timemask_meta.csv')
train_df3 = pd.read_csv('../csvs/spec_augs/spec_64_train_timewrap_meta.csv')
train_df4 = pd.read_csv('../csvs/spec_augs/spec_64_train_freqmask_meta.csv')
train_df = pd.concat([train_df1, train_df2, train_df3, train_df4], ignore_index=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = pd.read_csv('../csvs/spectrograms/spec_64_test_meta_16000.csv')
test_df = test_df.sample(frac=1).reset_index(drop=True)

from utils.dataloader import vit_load_data, class_weight_calc

# get x,y size of image using a random sample
counts = train_df['Label'].value_counts()
val_split = 0.15
datagen = ImageDataGenerator(validation_split=val_split)
x_size = 64
batchsize = 256
# Read images as 256x320 (expand by 7) to comply with patch rule: img_height*img_width % patchsize == 0
print("Load Data")
train_data, val_data, train_size, val_size = vit_load_data(datagen, train_df=train_df, val_split=val_split,
                                                           batchsize=batchsize, expanded_dim=320,
                                                           mode='training', test_df=None)

class_weights = class_weight_calc(train_df, class_mode='sklearn')
#Vit Hyper parameters
projection_dim = 128
transformer_units = [128, ]
patchsize = 16  # Split 256x320 image to  patchsizeXpatchsize patch images
patchnum = (x_size * 320) // (patchsize ** 2)  # as explained in original ViT paper
transformer_layers = 12
num_heads = 4
diag_attn_mask = 1 - tf.eye(patchnum)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
mlp_head_units = [64]
initializer = tf.keras.initializers.he_normal()
reg = None
from utils.create_model import vit_model
model = vit_model(x_size, 320, 1, initializer, reg, patchsize, patchnum, projection_dim,
                  transformer_layers, num_heads, diag_attn_mask, mlp_head_units,
                  logits=False, final_dense=True)

# total parameter count
initial_learning_rate = 0.0004
# Set callbacks
callback = []
save_best_callback = ModelCheckpoint(f'..../vision_transformers/weights/best_weights_vit.tf',
                                                        save_best_only=True, verbose=1)
callback.append(save_best_callback)
early_stop_callback = EarlyStopping(patience=6, restore_best_weights=True, verbose=1)
callback.append(early_stop_callback)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.93,
                              min_lr=0, mode='min', cooldown=0, verbose=1)
opt = tf.keras.optimizers.Nadam(learning_rate=initial_learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
callback_list = [callback, reduce_lr]
from utils.train_model import training
epochs = 50
model, hist = training(model, opt, loss, callback_list, class_weights, batchsize, train_size, val_size,
                       epochs, train_data, val_data)
model.save('../vision_transformers/vit_models/vit')
model.save_weights('../vision_transformers/weights/vit_weights.h5')

datagen = ImageDataGenerator()
test_it, test_data_size = vit_load_data(datagen, train_df=None, test_df=test_df, val_split=None,
                                        batchsize=batchsize, expanded_dim=320, mode='test')
score = model.evaluate(test_it, verbose=1)

print("Test loss is:", score[0])
print("Test acc is:", score[1])

print("Predicting")
y_pred = model.predict(test_it, steps=-(-test_data_size//batchsize), batch_size=batchsize, verbose=1)
# revert to decimal class representation
y_pred = np.argmax(y_pred, axis=1)
from utils.metrics import confmatrixplot, metric_plot, report_f1score

# Plot Metrics
confmatrixplot(test_df, y_pred, Filepath='../vision_transformers/metrics/conf_matrix.svg')
metric_plot(hist, Filepath='../vision_transformers/metrics/metricplot.svg')


from utils.mylogger import logger
params = model.count_params()
min_val_loss = round(early_stop_callback.best, 5)
best_epoch = early_stop_callback.best_epoch
desc = 'valsplit={},{},reg={},batch={}_lr={},patchsize={},numheads={},' \
       'projectiondim={},transformerunits={},mlphead{},'.format(val_split, opt, reg, batchsize,
                                                                initial_learning_rate, patchsize, num_heads,
                                                                projection_dim, transformer_units,
                                                                mlp_head_units)
print("Logging Initiated...")
name = 'transformer_model_run_10'

logger(name, params, round(hist.history['accuracy'][best_epoch], 4), round(hist.history['loss'][best_epoch], 5),
       round(hist.history['val_accuracy'][best_epoch], 4), min_val_loss, round(score[1], 4),
       round(score[0], 4), best_epoch, len(hist.history['loss']),
       round(report_f1score(test_df, y_pred), 4), desc)
print("Logging Complete...")
print("End of file...")