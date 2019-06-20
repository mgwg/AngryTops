import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.custom_loss import *

n_features_input = 6
n_target_features = 6

def base_model(**kwargs):
    """
    The base model. Consists of a since LSTM layer of 30 nodes sandwiched
    between two dense layers
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    learn_rate = kwargs["learn_rate"]
    lstm_size = 30
    if 'lstm_size' in kwargs.keys():
        lstm_size = int(kwargs['lstm_size'])
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def batchnorm_model(**kwargs):
    """
    The base model. Consists of a since LSTM layer of 30 nodes sandwiched
    between two dense layers
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    learn_rate = kwargs["learn_rate"]
    lstm_size = 30
    if 'lstm_size' in kwargs.keys():
        lstm_size = int(kwargs['lstm_size'])
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def double_lstm(**kwargs):
    """
    Create a RNN w/ 3 regularized LSTM layers, sandwiched between Batch
    Normalizations and Dense Layers
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    learn_rate = kwargs["learn_rate"]
    lstm0, lstm1 = 50, 50
    if 'lstm0' in kwargs.keys(): lstm0 = int(kwargs['lstm0'])
    if 'lstm1' in kwargs.keys(): lstm1 = int(kwargs['lstm1'])
    reg_weight = kwargs['reg_weight']
    rec_weight = kwargs['rec_weight']
    bias_weight = kwargs["bias_weight"]
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(lstm0, return_sequences=True, kernel_regularizer=l2(reg_weight),
                    recurrent_regularizer=l2(rec_weight), bias_regularizer=l2(bias_weight)))
    model.add(LSTM(lstm1, return_sequences=True, kernel_regularizer=l2(reg_weight),
                    recurrent_regularizer=l2(rec_weight), bias_regularizer=l2(bias_weight)))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def triple_lstm(**kwargs):
    """
    Create a RNN w/ 3 regularized LSTM layers, sandwiched between Batch
    Normalizations and Dense Layers
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    learn_rate = kwargs["learn_rate"]
    lstm0, lstm1, lstm2 = 50, 50, 50
    if 'lstm0' in kwargs.keys(): lstm0 = int(kwargs['lstm0'])
    if 'lstm1' in kwargs.keys(): lstm1 = int(kwargs['lstm1'])
    if 'lstm2' in kwargs.keys(): lstm0 = int(kwargs['lstm2'])
    reg_weight = kwargs['reg_weight']
    rec_weight = kwargs['rec_weight']
    bias_weight = kwargs["bias_weight"]
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(lstm0, return_sequences=True, kernel_regularizer=l2(reg_weight),
                    recurrent_regularizer=l2(rec_weight), bias_regularizer=l2(bias_weight)))
    model.add(LSTM(lstm1, return_sequences=True, kernel_regularizer=l2(reg_weight),
                    recurrent_regularizer=l2(rec_weight), bias_regularizer=l2(bias_weight)))
    model.add(LSTM(lstm2, return_sequences=True, kernel_regularizer=l2(reg_weight),
                    recurrent_regularizer=l2(rec_weight), bias_regularizer=l2(bias_weight)))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model_multi(**kwargs):
    """Seperate the inputs for jets and leps. Idea: Apply an LSTM layer to
    correctly identify the order of the jets. Then combine with lept + met
    information to obtain the final state objects. Uses only a single LSTM layer
    in the beginning for jets."""
    learn_rate = kwargs["learn_rate"]
    lstm_size = kwargs['lstm_size']
    dense_size = kwargs['dense_size']
    dense_act1 = 'linear'
    dense_act2 = 'relu'
    reg_weight = 0.0
    rec_weight = 0.0
    dense1_weight = 0.0
    if 'reg_weight' in kwargs.keys(): reg_weight = kwargs['reg_weight']
    if 'rec_weight' in kwargs.keys(): rec_weight = kwargs['rec_weight']
    if 'bias_weight' in kwargs.keys():bias_weight = kwargs["bias_weight"]
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']
    if 'dense_act2' in kwargs.keys(): dense_act2 = kwargs['dense_act2']
    if 'dense1_weight' in kwargs.keys(): dense_act2 = kwargs['dense1_weight']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(lstm_size, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(lstm_size//2, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(lstm_size//4, return_sequences=False,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)

    #x_jets = Reshape(target_shape=(5*lstm_size,))(x_jets)
    x_jets = Dense(dense_size, activation=dense_act1,
                   kernel_regularizer=l2(dense1_weight))(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(25, activation=dense_act2)(combined)
    final = Dense(18, activation='linear')(combined)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]
    lstm_size = kwargs['lstm_size']
    dense1 = kwargs['dense1']
    dense_act1 = 'relu'
    reg_weight = 0.0
    rec_weight = 0.0
    if 'reg_weight' in kwargs.keys(): reg_weight = kwargs['reg_weight']
    if 'rec_weight' in kwargs.keys(): rec_weight = kwargs['rec_weight']
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=False,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = Dense(30, activation='relu')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(dense1, activation=dense_act1)(combined)
    final = Dense(18, activation='elu')(final)
    final = Reshape(target_shape=(6,3))(final)
    final = Dense(3, activation="linear")(final)
    final = Dense(3, activation="linear")(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi2(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]
    dense_act1 = 'relu'
    reg_weight = 0.0
    rec_weight = 0.0
    if 'reg_weight' in kwargs.keys(): reg_weight = kwargs['reg_weight']
    if 'rec_weight' in kwargs.keys(): rec_weight = kwargs['rec_weight']
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(40, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(30, return_sequences=False,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = Dense(20, activation='relu')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(18, activation='tanh')(combined)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def single_var(**kwargs):
    """Predicts only ONE output variable"""
    learn_rate = kwargs["learn_rate"]
    lstm_size = kwargs['lstm_size']
    dense_size = kwargs['dense_size']
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    dense_act1 = 'linear'
    dense_act2 = 'relu'
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']
    if 'dense_act2' in kwargs.keys(): dense_act2 = kwargs['dense_act2']

    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True)(x_jets)
    x_jets = LSTM(30, return_sequences=True)(x_jets)
    x_jets = Reshape(target_shape=(5*30,))(x_jets)
    x_jets = Dense(dense_size, activation=dense_act1)(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(15, activation=dense_act2)(combined)
    final = Dense(5, activation="elu")(final)
    final = Dense(1, activation='linear')(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

################################################################################
# List of all models
models = {'base_model':base_model, 'batchnorm_model':batchnorm_model,
          'double_lstm':double_lstm, 'triple_lstm':triple_lstm,
          'model_multi':model_multi, 'dense_multi':dense_multi,
          'single_var':single_var, 'dense_multi2':dense_multi2}
################################################################################

if __name__ == "__main__":
    print("Compiled")
