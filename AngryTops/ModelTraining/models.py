import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from keras.layers.advanced_activations import LeakyReLU
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.single_output_models import *
from AngryTops.ModelTraining.cnn import cnn_models

def stacked_LSTM0(**kwargs):
    """A denser version of model_multi"""
    model = keras.models.Sequential()
    model.add(Dense(216, input_shape=(36,)))
    model.add(Reshape(target_shape=(6,36)))
    model.add(LSTM(108, return_sequences=True))
    #model.add(TimeDistributed(Dense(108, activation='tanh')))
    model.add(LSTM(72, return_sequences=True))
    #model.add(TimeDistributed(Dense(72, activation='tanh')))
    model.add(LSTM(36, return_sequences=True))
    #model.add(TimeDistributed(Dense(36, activation='tanh')))
    model.add(LSTM(18, return_sequences=True))
    #model.add(TimeDistributed(Dense(18, activation='tanh')))
    model.add(LSTM(3, return_sequences=True))
    #model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(10e-5, decay=0.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def stacked_LSTM1(**kwargs):
    """A denser version of model_multi"""
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
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def bidirectional_LSTM0(**kwargs):
    """A denser version of model_multi"""
    config = {'size1': 800.0, 'size2': 54.0}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    # Initially, due to typo, size1 = size2
    model.add(TimeDistributed(Dense(int(config['size1']), activation='tanh')))
    model.add(Bidirectional(LSTM(int(config['size2']), return_sequences=True)))
    model.add(TimeDistributed(Dense(27, activation='tanh')))
    model.add(TimeDistributed(Dense(9, activation='tanh')))
    model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(10e-5, decay=0.)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def bidirectional_LSTM1(**kwargs):
    """A denser version of model_multi"""
    config = {'size1': 800.0, 'size2': 100.0}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    # Initially, due to typo, size1 = size2
    model.add(TimeDistributed(Dense(int(config['size1']), activation='tanh')))
    model.add(Bidirectional(LSTM(int(config['size2']), return_sequences=True)))
    model.add(TimeDistributed(Dense(81, activation='tanh')))
    model.add(TimeDistributed(Dense(27, activation='tanh')))
    model.add(TimeDistributed(Dense(9, activation='tanh')))
    model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(10e-4, decay=0.)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def multiinput_BDLSTM(**kwargs):
    """A multi-input BDLSTM that first runs the Event level information through
    a DNN block and then concatenates w/ jet + lepton information and runs it
    through a BDLSTM block"""

    # Model Inputs
    input_events = Input(shape = (12,), name="input_events")
    input_features = Input(shape=(30,), name="input_features")

    # Event Information
    x_events = Dense(36, activation='tanh')(input_events)
    x_events = Dense(108, activation='tanh')(x_events)
    x_events = Dense(216, activation='tanh')(x_events)
    x_events = Dense(108, activation='tanh')(x_events)
    x_events = Dense(30, activation='tanh')(x_events)
    x_events = Reshape(target_shape=(6,5))(x_events)
    x_events = keras.Model(inputs=input_events, outputs=x_events)

    # Feature Information
    x_features = Reshape(target_shape=(6,5))(input_features)
    x_features = keras.Model(inputs=input_features, outputs=x_features)

    # BDLSTM BLOCK
    combined = concatenate([x_events.output, x_features.output], axis=-1)
    final = TimeDistributed(Dense(486, activation='tanh'))(combined)
    final = Bidirectional(LSTM(81, return_sequences=True))(final)
    final = TimeDistributed(Dense(81, activation='tanh'))(final)
    final = TimeDistributed(Dense(27, activation='tanh'))(final)
    final = TimeDistributed(Dense(9, activation='tanh'))(final)
    final = TimeDistributed(Dense(3, activation='tanh'))(final)
    model = keras.Model(inputs=[x_events.input, x_features.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(10e-5, decay=0.)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

################################################################################
# List of all models
models = {'stacked_LSTM0':stacked_LSTM0,
          'stacked_LSTM1':stacked_LSTM1,
          'bidirectional_LSTM0':bidirectional_LSTM0,
          'bidirectional_LSTM1':bidirectional_LSTM1,
          'multiinput_BDLSTM':multiinput_BDLSTM}

for key, constructor in single_models.items():
    models[key] = constructor

for key, constructor in cnn_models.items():
    models[key] = constructor

################################################################################

if __name__ == "__main__":
    model = stacked_LSTM1()
    print(model.summary())
