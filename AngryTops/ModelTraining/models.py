"""
Model architectures
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.single_output_models import *
from AngryTops.ModelTraining.cnn import cnn_models
from AngryTops.ModelTraining.custom_loss import *


def BDLSTM_model(metrics, losses, **kwargs):
    """
    A denser version of model_multi. For this case, we recommend the parameter
    input_size to be in kwargs. The valid input sizes are 24 or 36. If no
    input_size parameter is given, will default to an input size of 32.
    """
    # Model customization
    loss_fn = 'mse'
    if "custom_loss" in kwargs.keys(): loss_fn = losses[kwargs["custom_loss"]]
    input_size = 36
    n_part_output = 6
    target_shape = (6,6)
    if "input_size" in kwargs.keys():
        input_size = int(kwargs["input_size"])
        assert input_size == 36 or input_size == 24, "Invalid model input size"
    if input_size == 24:
        target_shape = (6,4)
        n_part_output = 2

    config = {'act1': 'relu', 'act2': 'relu', 'act3': 'elu',
              'act4': 'relu', 'size1': 440, 'size2': 44, 'size3':44, 'size4': 320,
              'size5': 90, 'size6': 30}

    # Model architecture
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=target_shape, input_shape=(input_size,)))
    model.add(TimeDistributed(Dense(int(config['size1']), activation=config['act1'])))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(int(config['size2']), return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(int(config['size3']), return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(int(config['size3']), return_sequences=True)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(int(config['size4']), activation=config['act2'])))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(int(config['size5']), activation=config['act3'])))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(int(config['size6']), activation=config['act3'])))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))
    model.add(TimeDistributed(Dense(3, activation='linear')))

    optimizer = tf.keras.optimizers.Adam(10e-4, decay=0.)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model

def cnn_model(metrics, losses, **kwargs):
    """A simple convolutional network model"""
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,), activation='relu'))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(18))
    model.add(Reshape(target_shape=(6,3)))

    optimizer = tf.keras.optimizers.Adam(10e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def LSTM_model(metrics, losses, **kwargs):
    """A denser version of model_multi"""
    loss_fn = 'mse'
    if "custom_loss" in kwargs.keys(): loss_fn = losses[kwargs["custom_loss"]]
    if "weights" in kwargs.keys(): weights = losses[kwargs["custom_loss"]]

    config = {'size1': 32, 'size2': 128, 'size3': 128, 'size4': 64, 'size5': 32}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    model.add(TimeDistributed(Dense(int(config['size1']), activation='tanh')))
    model.add(LSTM(int(config['size2']), return_sequences=True))
    #model.add(TimeDistributed(Dense(108, activation='tanh')))
    model.add(LSTM(int(config['size3']), return_sequences=True))
    #model.add(TimeDistributed(Dense(72, activation='tanh')))
    model.add(LSTM(int(config['size4']), return_sequences=True))
    #model.add(TimeDistributed(Dense(36, activation='tanh')))
    model.add(LSTM(int(config['size5']), return_sequences=True))
    #model.add(TimeDistributed(Dense(18, activation='tanh')))
    model.add(LSTM(3, return_sequences=True))
    #model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(0.0008965229699400112)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model

# List of all models
models = {'cnn_model':cnn_model,
          'LSTM_model':LSTM_model,
          'BDLSTM_model':BDLSTM_model}

for key, constructor in single_models.items():
    models[key] = constructor

for key, constructor in cnn_models.items():
    models[key] = constructor

################################################################################

if __name__ == "__main__":
    model = BDLSTM_model(metrics, losses, input_size=24)
    print(model.summary())
