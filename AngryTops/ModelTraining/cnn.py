import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.single_output_models import *


def cnn1(**kwargs):
    """A simple convolutional network model"""
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(10e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def cnn2(**kwargs):
    """A simple convolutional network model"""
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,), activation='relu'))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(18))
    model.add(Reshape(target_shape=(6,3)))

    optimizer = tf.keras.optimizers.Adam(10e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def cnn3(**kwargs):
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

def cnn2_hypopt(**kwargs):
    """A simple convolutional network model"""
    config = {'act1': 'tanh', 'act2': 'tanh', 'act3': 'tanh',
    'learn_rate': 0.0007631163113143247, 'size1': 127.0, 'size2': 158.0,
    'size3': 181.0}
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,), activation='relu'))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(int(config['size1']), activation=config['act1']))
    model.add(Dense(int(config['size2']), activation=config['act2']))
    model.add(Dense(int(config['size3']), activation=config['act3']))
    model.add(Dense(18))
    model.add(Reshape(target_shape=(6,3)))

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

cnn_models = {'cnn1': cnn1, 'cnn2': cnn2, 'cnn3': cnn3, 'cnn2_hypopt':cnn2_hypopt}

if __name__ == "__main__":
    model = cnn3()
    print(model.summary())
