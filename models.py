import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2


from features import *

n_features_input = 6
n_target_features = 6

def model0():
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
    model.add(Flatten())
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

################################################################################

def model1():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


################################################################################

def model2():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

################################################################################

def model3():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model4():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model5():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model6():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model7():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model8():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model9():
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

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def model10():
    """Seperate the inputs for jets and leps"""
    input_jets = Input(shape = (30,), name="input_jets")
    input_lep = Input(shape=(6,), name="input_lep")

    # Lepton Branch
    x_lep = Dense(6, activation='linear')(input_lep)
    x_lep = Reshape(target_shape=(1,6))(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Jets Branch
    x_jets = Reshape(target_shape=(5,6))(input_jets)
    x_jets = LSTM(6, return_sequences=True)(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = LSTM(4, return_sequences=True)(combined)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model




################################################################################
# List of all models
models = [model0, model1, model2, model3, model4, model5, model6, model7,
            model8, model9, model10]
################################################################################

if __name__ == "__main__":
    for model in models:
        model()
