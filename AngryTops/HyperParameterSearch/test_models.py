from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow import keras
import tensorflow as tf

def test_model0(config):
    print("INPUT")
    print("learn_rate: {0}\nsize1: {1}\nsize2: {2}\nsize3: {3}\nsize4: {4}\n\
          size5 {5}\nsize6: {6}\nsize7: {7}\nact1: {8}\nact2: {9}\nact3: {10}\n\
          act4: {11}\nreg_weight: {12}\nrec_weight: {13}".format(config['learn_rate'], config['size1'], \
          config['size2'], config['size3'], config['size4'], config['size5'], config['size6'], config['size7'], \
          config['act1'], config['act2'], config['act3'], config['act4'], config['reg_weight'], config['rec_weight']))

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

def test_model1(config):

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

def test_model2(config):
    """A denser version of model_multi"""
    kernel_reg1 = config['kernel_reg1']
    kernel_reg2 = config['kernel_reg2']
    kernel_reg3 = config['kernel_reg3']
    kernel_reg4 = config['kernel_reg4']
    kernel_reg5 = config['kernel_reg5']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(config['size1']), return_sequences=True, kernel_regularizer=l2(kernel_reg1))(x_jets)
    x_jets = LSTM(int(config['size2']), return_sequences=False, kernel_regularizer=l2(kernel_reg2))(x_jets)
    x_jets = Dense(int(config['size3']), activation=config['act1'], kernel_regularizer=l2(kernel_reg3))(x_jets)
    x_jets = Dense(int(config['size4']), activation=config['act2'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(config['size5']), activation=config['act3'])(input_lep)
    x_lep = Dense(int(config['size6']), activation=config['act4'])(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = BatchNormalization()(combined)
    final = Dense(int(config['size7']), activation=config['act5'], kernel_regularizer=l2(kernel_reg4))(final)
    final = Dense(int(config['size8']), activation=config['act6'], kernel_regularizer=l2(kernel_reg5))(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def test_model3(config):
    """A denser version of model_multi"""
    kernel_reg1 = config['kernel_reg1']
    kernel_reg2 = config['kernel_reg2']
    kernel_reg3 = config['kernel_reg3']
    kernel_reg4 = config['kernel_reg4']
    kernel_reg5 = config['kernel_reg5']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")

    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = Dense(int(config['size1']), activation=config['act1'], kernel_regularizer=l2(kernel_reg1))(x_jets)
    x_jets = LSTM(int(config['size2']), return_sequences=True, kernel_regularizer=l2(kernel_reg2))(x_jets)
    x_jets = LSTM(int(config['size3']), return_sequences=False, kernel_regularizer=l2(kernel_reg3))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = Dense(int(config['size4']), activation=config['act2'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(config['size5']), activation=config['act3'], kernel_regularizer=l2(kernel_reg4))(input_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = Dense(int(config['size6']), activation=config['act4'])(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(config['size7']), activation=config['act5'], kernel_regularizer=l2(kernel_reg5))(combined)
    final = Dense(18)(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

def cnn_model1(config):
    """A simple convolutional network model"""
    conv2d_tuple1 = (int(config['conv2d_tuple1x']), int(config['conv2d_tuple1y']))
    conv2d_tuple2 = (int(config['conv2d_tuple2x']), int(config['conv2d_tuple2y']))
    conv2d_tuple3 = (int(config['conv2d_tuple2x']), int(config['conv2d_tuple2y']))
    maxp_tuple1 = (int(config['maxp1x']), int(config['maxp1y']))
    maxp_tuple2 = (int(config['maxp2x']), int(config['maxp2y']))
    maxp_tuple3 = (int(config['maxp3x']), int(config['maxp3y']))
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,)))
    model.add(LeakyReLU(alpha=config['alpha1']))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(int(config['size1']), conv2d_tuple1, activation=config['act1'], padding="same"))
    model.add(MaxPooling2D(maxp_tuple1))
    model.add(Conv2D(int(config['size2']), conv2d_tuple2, activation=config['act2'], padding="same"))
    model.add(MaxPooling2D(maxp_tuple2))
    model.add(Conv2D(int(config['size3']), conv2d_tuple3, activation=config['act3'], padding="same"))
    model.add(MaxPooling2D(maxp_tuple3))
    model.add(Flatten())
    model.add(Dense(int(config['size4'])))
    model.add(LeakyReLU(alpha=config['alpha2']))
    model.add(Dense(24))
    model.add(Reshape(target_shape=(6,4)))

    optimizer = tf.keras.optimizers.Adam(10e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

test_models = {'test_model0': test_model0, 'test_model1': test_model1,
               'test_model2': test_model2, 'test_model3': test_model3}
