import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.custom_loss import *

def single1(**kwargs):
    """Predicts only ONE output variable"""
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    if 'dense_act1' in kwargs.keys(): dense_act1 = kwargs['dense_act1']
    if 'dense_act2' in kwargs.keys(): dense_act2 = kwargs['dense_act2']

    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True)(x_jets)
    x_jets = LSTM(30, return_sequences=True)(x_jets)
    x_jets = LSTM(20, return_sequences=False)(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = Dense(20, activation='tanh')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(15, activation='relu')(combined)
    final = Dense(5, activation="elu")(final)
    final = Dense(1, activation='linear')(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(10e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def single2(**kwargs):
    """Predicts only ONE output variable. Uses only dense layers"""
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")

    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True)(x_jets)
    x_jets = LSTM(40, return_sequences=True)(x_jets)
    x_jets = LSTM(30, return_sequences=False)(x_jets)
    x_jets = Dense(25, activation='relu')(x_jets)
    x_jets = Dense(20, activation='relu')(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu')(input_lep)
    x_lep = Dense(15, activation='relu')(x_lep)
    x_lep = Dense(10, activation='relu')(x_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(15, activation='relu')(combined)
    final = Dense(5, activation="elu")(final)
    final = Dense(1, activation='linear')(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)
    optimizer = tf.keras.optimizers.Adam(10e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

def single3(**kwargs):
    """Predicts only ONE output variable. Uses only dense layers"""
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")

    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = Dense(50, activation='relu')(x_jets)
    x_jets = Dense(40, activation='relu')(x_jets)
    x_jets = LSTM(30, return_sequences=False)(x_jets)
    x_jets = Dense(25, activation='relu')(x_jets)
    x_jets = Dense(20, activation='relu')(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu')(input_lep)
    x_lep = Dense(15, activation='relu')(x_lep)
    x_lep = Dense(10, activation='relu')(x_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(15, activation='relu')(combined)
    final = Dense(5, activation="elu")(final)
    final = Dense(1, activation='linear')(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)
    optimizer = tf.keras.optimizers.Adam(10e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

################################################################################
# List of all models
single_models = {'single_var':single_var}
################################################################################
