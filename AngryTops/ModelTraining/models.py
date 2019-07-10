import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from keras.layers.advanced_activations import LeakyReLU
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.single_output_models import *
from AngryTops.ModelTraining.cnn import cnn_models


def dense_multi1(**kwargs):
    """A denser version of model_multi"""
    dense_act1 = 'relu'
    reg_weight = 0.0
    rec_weight = 0.0
    if 'reg_weight' in kwargs.keys(): reg_weight = kwargs['reg_weight']
    if 'rec_weight' in kwargs.keys(): rec_weight = kwargs['rec_weight']
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']
    learn_rate = kwargs['learn_rate']

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
    final = Dense(40, activation=dense_act1)(combined)
    final = Dense(18, activation='elu')(final)
    final = Dense(18, activation='elu')(final)
    final = Dense(18, activation='elu')(final)
    final = Reshape(target_shape=(6,3))(final)
    final = Dense(3, activation="linear")(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi2(**kwargs):
    """A denser version of model_multi"""
    reg_weight = 0.0
    rec_weight = 0.0
    if 'reg_weight' in kwargs.keys(): reg_weight = kwargs['reg_weight']
    if 'rec_weight' in kwargs.keys(): rec_weight = kwargs['rec_weight']
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']
    learn_rate = kwargs['learn_rate']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(25, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(12, return_sequences=False,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = Dense(20, activation='tanh')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(10, activation='tanh')(input_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(50, activation='tanh')(combined)
    final = Dense(25, activation='tanh')(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model


def dense_multi3(**kwargs):
    config = {'act1': 'tanh', 'act2': 'relu', 'act3': 'relu', 'act4': 'tanh',
    'learn_rate': 0.0008838279551810702, 'rec_weight': 0.7514254449992382,
    'reg_weight': 0.28301311919728966, 'size1': 19.0, 'size2': 77.0,
    'size3': 194.0, 'size4': 88.0, 'size5': 64.0, 'size6': 197.0, 'size7': 111.0}

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(config['size1']), return_sequences=True,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = LSTM(int(config['size2']), return_sequences=True,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = LSTM(int(config['size3']), return_sequences=False,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = Dense(int(config['size4']), activation=config['act1'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(config['size5']), activation=config['act2'])(input_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(config['size6']), activation=config['act3'])(combined)
    final = Dense(int(config['size7']), activation=config['act4'])(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi4(**kwargs):

    config = {'act1': 'elu', 'act2': 'relu', 'act3': 'elu', 'act4': 'relu',
    'act5': 'tanh', 'learn_rate': 0.0008238245582829519,
    'rec_weight': 0.012269177904945305, 'reg_weight': 0.7402757786314242,
    'size1': 10.0, 'size2': 133.0, 'size3': 137.0, 'size4': 116.0, 'size5': 171.0}

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(config['size1']), return_sequences=False,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = Dense(int(config['size2']), activation=config['act1'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(config['size3']), activation=config['act2'])(combined)
    final = Dense(int(config['size4']), activation=config['act3'])(final)
    final = Dense(int(config['size5']), activation=config['act4'])(final)
    final = Dense(18, activation=config['act5'])(final)
    final = Reshape(target_shape=(6,3))(final)
    final = Dense(3, activation="linear")(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi5(**kwargs):
    """A denser version of model_multi. Uses TimeDistributed layers between
    LSTM laters"""
    reg_weight = 0.0
    rec_weight = 0.0
    learn_rate = kwargs['learn_rate']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = TimeDistributed(Dense(50, activation='relu'))(x_jets)
    x_jets = LSTM(25, return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = TimeDistributed(Dense(25, activation='relu'))(x_jets)
    x_jets = LSTM(12, return_sequences=False,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(10, activation='tanh')(input_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(50, activation='tanh')(combined)
    final = Dense(25, activation='tanh')(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

################################################################################
# List of all models
models = {'dense_multi1':dense_multi1,
          'dense_multi2':dense_multi2,'dense_multi3':dense_multi3,
          'dense_multi4':dense_multi4, 'dense_multi5':dense_multi5}

for key, constructor in single_models.items():
    models[key] = constructor

for key, constructor in cnn_models.items():
    models[key] = constructor

################################################################################

if __name__ == "__main__":
    model = dense_multi5(learn_rate=10e-5)
    print(model.summary())
