import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from keras_adabound import AdaBound

test_df = pd.read_csv('../csvs/spectrograms/spec_64_test_meta_16000.csv')
test_df = test_df.sample(frac=1).reset_index(drop=True)
datagen = ImageDataGenerator()
batchsize = 64
sample = plt.imread(test_df['Filepath'][0])
x_size, y_size = np.shape(sample)
del sample
test_data = datagen.flow_from_dataframe(dataframe=test_df, target_size=(x_size, y_size),
                                        x_col='Filepath', y_col='Label', class_mode="categorical",
                                        batch_size=batchsize, shuffle=False, color_mode="grayscale")

labels = test_data.classes
char_labels = list(test_data.class_indices)
print("Models loading...")
name = 'ensemble of diff(diff_kernels) and same kernel(mymodel2_7th)'
model = tf.keras.models.load_model('../2d_convs/diff_kernel_models/diff_kernels', compile=False)
model._name = "first_model"
model2 = tf.keras.models.load_model('../2d_convs/same_kernel_models/mymodel2_7thrun', compile=False)
model2._name = "second_model"
models = [model, model2]
# New input size
model_input = tf.keras.Input(shape=(x_size, y_size, 1))
# New output size
model_outputs = [model(model_input) for model in models]
# Ensemble output layer
from utils.weighted_average import WeightedAverageLayer
#  WeightedAverageLayer(w1, w2, name='weighted_average')
# w1 = 0.4
# w2 = 0.6
ensemble_out = tf.keras.layers.Average(name='average_outputs')(model_outputs)
# Create model
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_out)
ensemble_model.summary()
tf.keras.utils.plot_model(ensemble_model, show_layer_activations=True)
opt = AdaBound(learning_rate=0.0004, final_lr=0.1, amsgrad=True)
ensemble_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = ensemble_model.evaluate(test_data)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Predicting")

y_pred = ensemble_model.predict(test_data, verbose=1)
# revert to decimal class representation
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_pred=y_pred, y_true=labels, target_names=char_labels, digits=4))
from utils.metrics import confmatrixplot
print("Calculating confusion matrix")
confmatrixplot(test_df, y_pred, '../ensemble_nets/confmatrices/confmatrix_{}.svg'.format(name))
plt.show()
print("Logging Initiated...")
# Details about current session, to use for logging purposes
params = ensemble_model.count_params()
from utils.metrics import report_f1score
from utils.mylogger import logger
f1score = report_f1score(test_df, y_pred)
desc = 'Ensemble{}_lr={}_batchsize={}'.format(opt, 0.0004, batchsize)
logger(name, params, '_', '_', '_', '_', round(score[1], 4),
       round(score[0], 4), '_', '_', round(f1score, 4), desc)
print("Logging Complete...")
print("End of file...")
