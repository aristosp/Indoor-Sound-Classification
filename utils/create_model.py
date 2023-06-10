from tensorflow.keras.layers import Dense, add, BatchNormalization, Dropout, Conv2D, Flatten,\
    MultiHeadAttention, LayerNormalization, Input, concatenate, Activation, average,\
    Conv1D, MaxPooling2D, Rescaling, GlobalMaxPool2D
import tensorflow as tf
from vit_utils import ShiftedPatchTokenization, MultiHeadAttentionLSA, PatchEncoder
from keras.models import Sequential
from keras.layers import Dense, add, BatchNormalization, Dropout, Conv2D, Flatten,\
    AveragePooling2D, GlobalMaxPooling2D, Input, concatenate, Activation, average,\
    Conv1D, MaxPooling2D, Permute, Reshape, Rescaling, MaxPooling1D
import tensorflow_addons as tfa



def conv1d(x_size, y_size, n_channels, grouping_num, initializer, regularizer, name, activation):
    """
    This function creates a 1D Convolution model
    :param x_size: Dimension of x_axis
    :param y_size: Dimension of y_axis
    :param n_channels: Whether the images to be used are in grayscale or RGB format
    :param grouping_num: Grouping parameter for group normalization layers
    :param initializer: Weight initializer function
    :param regularizer: Regularizer function
    :param name: Model name
    :param activation: Activation function to be used in every layer, except the output one
    :return: A model object
    """
    groups_a = grouping_num
    # Preprocessing layers
    # Reshape data to be in the following format:(x_size,y_size) from (x_size, y_size,1)
    # Permute the data from (n_mfcc,time) to (time,n_mfcc), in order to have 1D Convolution
    model_input = Input(shape=(x_size, y_size, n_channels), name='input_layer')
    reshape_layer = Reshape(target_shape=(x_size, y_size), name='reshape_layer')(model_input)
    # to rescale between [-1,1] use rescale=1./127.5, offset=-1
    # to rescale between [0,1] use rescale=1./255, offset=0
    rescale_layer = Rescaling(scale=1. / 255, offset=0, name='Rescale_layer')(reshape_layer)
    permute_layer = Permute((2, 1), name='permute_layer')(rescale_layer)
    # First layers of each side, a denotes one model, and b the other
    group1_a = tfa.layers.GroupNormalization(groups=groups_a, name='group1_a')(permute_layer)
    conv1d_1a = tfa.layers.WeightNormalization(
        Conv1D(32, kernel_size=3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer,
               name='conv1d_1a'))(group1_a)
    # maxpool_a = MaxPooling1D(name='maxpool_a')(conv1d_1a)
    # Second layers
    group2_a = tfa.layers.GroupNormalization(groups=groups_a, name='group2_a')(conv1d_1a)
    conv1d_2a = tfa.layers.WeightNormalization(
        Conv1D(64, kernel_size=3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer,
               name='conv1d_2a'))(group2_a)
    dropout1_a = Dropout(0.5, name='dropout1_a')(conv1d_2a)
    maxpool1_a = MaxPooling1D(name='maxpool1_a')(dropout1_a)

    # Third layers
    group3_a = tfa.layers.GroupNormalization(groups=groups_a, name='group3_a')(maxpool1_a)
    conv1d_3a = tfa.layers.WeightNormalization(
        Conv1D(128, kernel_size=3, activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1d_3a'))(group3_a)
    dropout2_a = Dropout(0.5, name='dropout2_a')(conv1d_3a)
    maxpool2_a = MaxPooling1D(name='maxpool2_a')(dropout2_a)

    # Fourth Layers
    group4_a = tfa.layers.GroupNormalization(groups=groups_a, name='group4_a')(maxpool2_a)
    conv1d_4a = tfa.layers.WeightNormalization(
        Conv1D(256, kernel_size=3, activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1d_4a'))(group4_a)
    dropout3_a = Dropout(0.5, name='dropout3_a')(conv1d_4a)
    maxpool3_a = MaxPooling1D(name='maxpool3_a')(dropout3_a)

    # Fifth layers
    group5_a = tfa.layers.GroupNormalization(groups=groups_a, name='group5_a')(maxpool3_a)
    conv1d_5a = tfa.layers.WeightNormalization(
        Conv1D(512, kernel_size=3, activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1d_5a'))(group5_a)
    dropout4_a = Dropout(0.5, name='dropout4_a')(conv1d_5a)
    maxpool4_a = MaxPooling1D(name='maxpool4_a')(dropout4_a)

    # Last group normalization and flattening
    group8_a = tfa.layers.GroupNormalization(groups=groups_a, name='group8_a')(maxpool4_a)
    flatten_a = Flatten(name='flatten_a')(group8_a)
    # Dense layers
    dropout6_a = Dropout(0.5, name='dropout6_a')(flatten_a)
    dense2_a = tfa.layers.WeightNormalization(
        Dense(128, activation=activation, kernel_initializer=initializer,
              kernel_regularizer=regularizer, name='dense2_a'))(dropout6_a)
    dropout7_a = Dropout(0.5, name='dropout7_a')(dense2_a)
    out_a = tfa.layers.WeightNormalization(
        Dense(9, activation='softmax', kernel_initializer=initializer, kernel_regularizer=regularizer,
              name='out_a'))(dropout7_a)
    model = tf.keras.models.Model(inputs=model_input, outputs=out_a, name=name)
    return model


# The following function creates a 2D Convolutional model with symmetrical kernels
def conv2d_same_kernels(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):
    """ This function returns a keras model object.
    :param name: model name
    :param x_size: image dimension
    :param y_size: image dimension
    :param n_channels: Whether images are in grayscale or rgb format
    :param initializer: kernel initializing function to use
    :param reg: Kernel regularizer function to use for generalization puprposes
    :param batchsize: Batchsize to be used
    :param logits: Define whether the model returns the logits or the probabilities
    :returns model: A keras model object
    """
    model = Sequential(name=name)
    # Input layer
    model.add(Rescaling(1 / 255, offset=0, name='rescaling', input_shape=(x_size, y_size, n_channels)))
    model.add(BatchNormalization(name='batchnorm1'))
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer=initializer, activation='gelu',
                     input_shape=(x_size, y_size, n_channels), padding='same', kernel_regularizer=reg, name='conv1d_1'))
    model.add(MaxPooling2D(name='maxpool1'))
    model.add(Dropout(0.4))
    # First Hidden Layer
    model.add(BatchNormalization(name='batchnorm2'))
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu',
                     name='conv1d_2'))
    model.add(MaxPooling2D(name='maxpool2'))
    model.add(Dropout(0.4))
    # Second Hidden Layer
    model.add(BatchNormalization(name='batchnorm3'))
    model.add(Conv2D(128, kernel_size=(3, 3), kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu',
                     name='conv1d_3'))
    model.add(MaxPooling2D(name='maxpool3'))
    model.add(Dropout(0.5, name='dropout2'))
    # Third Hidden Layer
    model.add(BatchNormalization(name='batchnorm4'))
    model.add(Conv2D(256, kernel_size=(5, 5), kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu',
                     name='conv1d_4'))
    model.add(MaxPooling2D(name='maxpool4'))

    # Output Layer
    model.add(GlobalMaxPooling2D(name='GlobalMaxPool'))
    model.add(BatchNormalization(name='batchnorm5'))
    model.add(Dense(64, kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu', name='dense1'))
    model.add(Dropout(0.5, name='dropout3'))
    if logits:
        model.add(Dense(9, kernel_regularizer=reg, kernel_initializer=initializer, name='output'))
    else:
        model.add(Dense(9, kernel_regularizer=reg, kernel_initializer=initializer, activation='softmax', name='output'))
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))

    return model


# Create a Conv2D model with asymmetrical kernels
def conv2d_diff_kernels(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):
    """ This function returns a keras model object.
    :param name: model name
    :param x_size: image dimension
    :param y_size: image dimension
    :param n_channels: Whether images are in grayscale or rgb format
    :param initializer: kernel initializing function to use
    :param reg: Kernel regularizer function to use for generalization puprposes
    :param batchsize: Batchsize to be used
    :param logits: Define whether the model returns the logits or the probabilities
    :returns model: A keras model object
    """
    # Input layers
    model_input = Input(shape=(x_size, y_size, n_channels), name='Input')
    rescale = Rescaling(1 / 255, offset=0, name='rescale')(model_input)

    # First layers
    batch_norm1 = BatchNormalization(name='batch1')(rescale)
    conv2d_1 = Conv2D(64, kernel_size=(5, 1), kernel_initializer=initializer, activation='gelu',
                      kernel_regularizer=reg, name='conv2d_1')(batch_norm1)
    dropout = Dropout(0.2, name='dropout')(conv2d_1)
    max_pool = MaxPooling2D(pool_size=(3, 1), name='maxpool')(dropout)

    # Second layers
    batch_norm2 = BatchNormalization(name='batch2')(max_pool)
    conv2d_2 = Conv2D(128, kernel_size=(10, 1), kernel_initializer=initializer, activation='gelu',
                      kernel_regularizer=reg, name='conv2d_2')(batch_norm2)

    # Third layers
    batch_norm3 = BatchNormalization(name='batch3')(conv2d_2)
    dropout1 = Dropout(0.3, name='dropout1')(batch_norm3)
    conv2d_3 = Conv2D(256, kernel_size=(1, 6), kernel_initializer=initializer, activation='gelu',
                      kernel_regularizer=reg, name='conv2d_3')(dropout1)
    max_pool2 = MaxPooling2D(pool_size=(1, 4), name='maxpool2')(conv2d_3)
    batch_norm4 = BatchNormalization(name='batch4')(max_pool2)

    Gap = GlobalMaxPool2D(name='gap')(batch_norm4)
    # Dense layers
    dense = Dense(128, activation='gelu', kernel_initializer=initializer, name='dense', kernel_regularizer=reg)(Gap)
    dropout = Dropout(0.5, name='dropout2')(dense)
    if logits:
        model_output = Dense(9, activation=None, kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)
    else:
        model_output = Dense(9, activation='softmax', kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)
    model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=name)
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))
    return model

# The following functions relate to the creation of a vision transformer model


def mlp(x, hidden_units, dropout_rate, kernel_initializer, kernel_regularizer):
    """This function returns a multi perceptron unit, with Dense and dropout.
    The activation function used is gelu.
    :param hidden_units: number of hidden units to create
    :param dropout_rate: the probability for dropout layers to use
    :param kernel_initializer: the randomly initialized weight function
    :param kernel_regularizer: whether to use regularization to increase generalization
    """
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu, kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)(x)
        x = Dropout(dropout_rate)(x)
    return x


def vit_model(x_size, y_size, n_channels, initializer, reg, patchsize, patchnum, projection_dim,
              transformer_layers, num_heads, mask, mlp_units, logits=False, final_dense=False):
    """ This function returns a keras model object.
    :param x_size: image dimension
    :param y_size: image dimension
    :param n_channels: Whether images are in grayscale or rgb format
    :param initializer: kernel initializing function to use
    :param reg: Kernel regularizer function to use for generalization puprposes
    :param patchsize: Dimensions of created patches. The patches' dimensions will be patchsize X patchsize
    :param patchnum: Number of patches to be created
    :param projection_dim: Dimension where the linear projection will happen
    :param transformer_layers: Number of transformer layers to be used
    :param num_heads: Number of heads for multi-head attention layers
    :param mask: The mask to be used
    :param mlp_units: Number of MLP units to be used before output
    :param logits: Define whether the model returns the logits or the probabilities
    :param final_dense: Whether dense layers before the output layer will be used
    :returns model: A keras model object
    """
    model_input = Input(shape=(x_size, y_size, n_channels), name='model_input')
    rescale_layer = Rescaling(scale=1./255, offset=0, name='Rescale_layer')(model_input)
    batch1 = BatchNormalization(name='batch1')(rescale_layer)
    # Patch creator
    (patch_layer, _) = ShiftedPatchTokenization(patch_size=patchsize, vanilla=False, image_size_h=x_size,
                                                image_size_w=y_size, num_patches=patchnum,
                                                projection_dim=projection_dim)(batch1)
    # Patch encoding
    encoding_layer = PatchEncoder(num_patches=patchnum, projection_dim=projection_dim)(patch_layer)
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoding_layer)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttentionLSA(num_heads=num_heads, key_dim=projection_dim,
                                                 dropout=0.5)(x1, x1, attention_mask=mask)
        # Skip connection 1.
        x2 = add(inputs=[attention_output, encoding_layer])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        # x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.5, kernel_initializer=initializer)
        # Skip connection 2.
        encoding_layer = add(inputs=[x3, x2])
    representation = LayerNormalization(epsilon=1e-6)(encoding_layer)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP and classify outputs.
    if final_dense:
        features = mlp(representation, hidden_units=mlp_units, dropout_rate=0.5, kernel_initializer=initializer, kernel_regularizer=reg)
        if logits:
            out = Dense(9, kernel_regularizer=reg, kernel_initializer=initializer)(features)
        else:
            out = Dense(9, activation='softmax', kernel_initializer=initializer, kernel_regularizer=reg)(features)
    else:
        if logits:
            out = Dense(9, kernel_regularizer=reg, kernel_initializer=initializer)(representation)
        else:
            out = Dense(9, activation='softmax', kernel_initializer=initializer, kernel_regularizer=reg)(representation)

    # Create the Keras model
    model = tf.keras.models.Model(inputs=model_input, outputs=out)
    return model
