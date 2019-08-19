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
    """A denser version of model_multi"""
    loss_fn = 'mse'
    if "custom_loss" in kwargs.keys(): loss_fn = losses[kwargs["custom_loss"]]

    config = {'act1': 'relu', 'act2': 'relu', 'act3': 'elu',
              'act4': 'relu', 'size1': 440, 'size2': 44, 'size3':44, 'size4': 320,
              'size5': 90, 'size6': 30}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    # Initially, due to typo, size1 = size2
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
    model.add(TimeDistributed(Dense(3, activation='linear')))

    optimizer = tf.keras.optimizers.Adam(10e-4, decay=0.)
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
    model = BDLSTM_model(metrics, losses)
    print(model.summary())
