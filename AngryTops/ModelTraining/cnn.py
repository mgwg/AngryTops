import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.custom_loss import *
from AngryTops.ModelTraining.single_output_models import *

def cnn1(**kwargs):
    """A simple convolutional network model"""
    model = keras.models.Sequential()
    model.add(Dense(128, input_shape=(36,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape(target_shape=(8,8,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
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

cnn_models = {'cnn1': cnn1, 'cnn2': cnn2}

if __name__ == "__main__":
    model = cnn2()
    print(model.summary())
