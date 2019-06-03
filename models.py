import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *


from features import *

n_features_input = 6
n_target_features = 6

def model0(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(SimpleRNN(30, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

################################################################################

def model1(learn_rate):
    """
    Create a simple RNN with L2 regularizers and dropout
    """
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,), kernel_regularizer=l2()))
    model.add(Dropout(0.2))
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(30, return_sequences=True, kernel_regularizer=l2()))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


################################################################################

def model2(learn_rate):
    """
    Create a simple RNN with L2 regularizers and dropout layers
    """
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Dropout(0.2))
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(30, return_sequences=True, kernel_regularizer=l2()))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

################################################################################

def model3(learn_rate):
    """
    Create a simple RNN with L2 regularizers and dropout layers
    """
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Dropout(0.2))
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(36, return_sequences=True))
    model.add(LSTM(36, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model4(learn_rate):
    """
    Create a simple RNN with L2 regularizers and dropout layers
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Dropout(0.1))
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(30, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(LSTM(30, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model5(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(30, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model6(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(SimpleRNN(30, return_sequences=True))
    model.add(SimpleRNN(30, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model7(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(SimpleRNN(100, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model8(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(SimpleRNN(100, return_sequences=True))
    model.add(SimpleRNN(100, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model9(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(Reshape(target_shape=(6,6)))
    model.add(SimpleRNN(30, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model10(learn_rate):
    """Seperate the inputs for jets and leps"""
    input_jets = Input(shape = (30,), name="input_jets")
    input_lep = Input(shape=(6,), name="input_lep")

    # Lepton Branch
    x_lep = Dense(10, activation='relu')(input_lep)
    x_lep = Dense(8, activation='relu')(input_lep)
    x_lep = Dense(8, activation='relu')(x_lep)
    x_lep = Reshape(target_shape=(1,8))(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Jets Branch
    x_jets = Reshape(target_shape=(5,6))(input_jets)
    x_jets = LSTM(10, return_sequences=True)(x_jets)
    x_jets = LSTM(8, return_sequences=True)(x_jets)
    x_jets = Dense(8, activation="relu")(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = LSTM(6, return_sequences=True)(combined)
    final = Dense(4, activation="linear")(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def model11(learn_rate):
    """Seperate the inputs for jets and leps"""
    input_jets = Input(shape = (30,), name="input_jets")
    input_lep = Input(shape=(6,), name="input_lep")

    # Lepton Branch
    x_lep = Dense(10, activation='relu')(input_lep)
    x_lep = Dense(10, activation='relu')(input_lep)
    x_lep = Dense(6, activation='relu')(x_lep)
    x_lep = Reshape(target_shape=(1,6))(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Jets Branch
    x_jets = Reshape(target_shape=(5,6))(input_jets)
    x_jets = LSTM(10, return_sequences=True)(x_jets)
    x_jets = LSTM(10, return_sequences=True)(x_jets)
    x_jets = Dense(6, activation="relu")(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = LSTM(6, return_sequences=True)(combined)
    final = Dense(4, activation="linear")(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def model12(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.RMSProp(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model13(learn_rate):
    """
    Create a simple RNN with one recurrent layer
    """
    # Questions:
    # 1. Originally there was a TimeDIstributed Layer. I think this was
    # unceccessary
    # The return_sequences=True argument ==> Not sure what this does
    # I simplified the model significantly => Reduced it to one recurrent layer
    model = keras.Sequential()
    model.add(Dense(36, activation='relu', input_shape=(36,)))
    model.add(BatchNormalization())
    model.add(Reshape(target_shape=(6,6)))
    model.add(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01),
                    recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01),
                    recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01),
                    recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


################################################################################
# List of all models
models = [model0, model1, model2, model3, model4, model5, model6, model7,
            model8, model9, model10, model11, model12, model13]
################################################################################

if __name__ == "__main__":
    for model in models:
        model()
