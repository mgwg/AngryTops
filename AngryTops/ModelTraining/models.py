import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import sys
from AngryTops.features import *
from AngryTops.ModelTraining.custom_loss import *
from AngryTops.ModelTraining.single_output_models import *

def dense_multi1(**kwargs):
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

def dense_multi3(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]
    dense_act1 = 'relu'
    reg_weight = 0.0
    rec_weight = 0.0

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = LSTM(40, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = LSTM(30, return_sequences=False, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = Dense(20, activation='relu')(x_jets)
    x_jets = Dense(20)(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(input_lep)
    x_lep = Dense(15, activation='relu', kernel_regularizer=l2(10e-5))(x_lep)
    x_lep = Dense(10, activation='linear')(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(30, activation='relu', kernel_regularizer=l2(10e-5))(combined)
    final = Dense(25, activation='relu', kernel_regularizer=l2(10e-5))(final)
    final = Dense(18, activation='relu', kernel_regularizer=l2(10e-5))(final)
    final = Dense(18)(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi4(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]
    dense_act1 = 'relu'
    reg_weight = 0.0
    rec_weight = 0.0

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(input_jets)
    x_jets = Reshape(target_shape=(5,4))(x_jets)
    x_jets = LSTM(50, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = LSTM(40, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = LSTM(30, return_sequences=False, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = Dense(20, activation='linear')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(input_lep)
    x_lep = Dense(15, activation='relu', kernel_regularizer=l2(10e-5))(x_lep)
    x_lep = Dense(10, activation='linear')(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(30, activation='relu', kernel_regularizer=l2(10e-5))(combined)
    final = Dense(25, activation='relu', kernel_regularizer=l2(10e-5))(final)
    final = Dense(18, activation='relu', kernel_regularizer=l2(10e-5))(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi5(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(input_jets)
    x_jets = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = Reshape(target_shape=(5,4))(x_jets)
    x_jets = LSTM(50, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = LSTM(40, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = LSTM(30, return_sequences=False, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = Dense(20, activation='linear')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu', kernel_regularizer=l2(10e-5))(input_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = Dense(15, activation='relu', kernel_regularizer=l2(10e-5))(x_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = Dense(10, activation='linear')(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(30, activation='relu', kernel_regularizer=l2(10e-5))(combined)
    final = Dense(25, activation='relu', kernel_regularizer=l2(10e-5))(final)
    final = Dense(18, activation='relu', kernel_regularizer=l2(10e-5))(final)
    final = Dense(18)(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi6(**kwargs):
    """A denser version of model_multi"""
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Dense(50, activation='relu', kernel_regularizer=l2(10e-5))(input_jets)
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(50, return_sequences=True, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = LSTM(30, return_sequences=False, kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = Dense(30, activation='relu', kernel_regularizer=l2(10e-5))(x_jets)
    x_jets = Dense(20, activation='linear')(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='elu', kernel_regularizer=l2(10e-5))(input_lep)
    x_lep = Dense(15, activation='elu', kernel_regularizer=l2(10e-5))(x_lep)
    x_lep = Dense(10, activation='linear')(input_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(30, activation='relu')(combined)
    final = Dense(25, activation='elu')(final)
    final = Dense(18, activation='elu')(final)
    final = Dense(18, activation="linear")(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(10e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi7(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = LSTM(50, return_sequences=True, kernel_regularizer=l2(0.005))(x_jets)
    x_jets = LSTM(30, return_sequences=False, kernel_regularizer=l2(0.005))(x_jets)
    x_jets = Dense(20, activation='relu', kernel_regularizer=l2(0.0001))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu', kernel_regularizer=l2(0.0001))(input_lep)
    x_lep = Dense(15, activation='relu', kernel_regularizer=l2(0.0001))(x_lep)
    x_lep = Dense(10, activation='linear')(x_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Reshape(target_shape=(6,5))(combined)
    final = LSTM(30, return_sequences=False, kernel_regularizer=l2(0.005))(final)
    final = Dense(25, activation='relu', kernel_regularizer=l2(0.0001))(final)
    final = Dense(18, activation='elu', kernel_regularizer=l2(0.0001))(final)
    final = Dense(18)(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def dense_multi8(**kwargs):
    """A denser version of model_multi"""
    learn_rate = kwargs["learn_rate"]

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = LSTM(50, return_sequences=True, kernel_regularizer=l2(0.001))(x_jets)
    x_jets = LSTM(30, return_sequences=False, kernel_regularizer=l2(0.0001))(x_jets)
    x_jets = Dense(20, activation='relu')(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(20, activation='relu', kernel_regularizer=l2(0.0001))(input_lep)
    x_lep = Dense(15, activation='relu', kernel_regularizer=l2(0.0001))(x_lep)
    x_lep = Dense(10, activation='linear')(x_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(30, activation='relu', kernel_regularizer=l2(0.0001))(combined)
    final = Dense(25, activation='relu', kernel_regularizer=l2(0.0001))(final)
    final = Dense(25, activation='relu', kernel_regularizer=l2(0.0001))(final)
    final = Dense(18, activation='elu', kernel_regularizer=l2(0.0001))(final)
    final = Dense(18)(final)
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
          'dense_multi4':dense_multi4,'dense_multi5':dense_multi5,
          'dense_multi6':dense_multi6,'dense_multi7':dense_multi7,
          'dense_multi8':dense_multi8}

for key, constructor in single_models.items():
    models[key] = constructor

################################################################################

if __name__ == "__main__":
    print("Compiled")
