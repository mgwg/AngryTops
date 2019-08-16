import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from AngryTops.features import *
from AngryTops.ModelTraining.single_output_models import *

EPOCHES = 100
BATCH_SIZE=32
(training_input, training_output), (testing_input, testing_output), \
(jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = \
get_input_output(input_filename="topreco_5dec3.csv", scaling='minmax',
rep="experimental", multi_input=False, sort_jets=False, particle="W_lep_cart")

# Make + Train Models
WLEP_model = BDLSTM_model()
cp_callback = ModelCheckpoint(".", monitor='val_loss', save_weights_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=0)
history = model.fit(training_input, training_output,  epochs=EPOCHES,
                    batch_size=BATCH_SIZE, validation_split=0.1,
                    callbacks = [early_stopping, cp_callback]
                    )
model.save('WLep_Model.h5')

# Use the WLEP model to update the training/testing data
(training_input, training_output), (testing_input, testing_output), \
(jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = \
get_input_output(input_filename="topreco_5dec3.csv", scaling='minmax',
rep="experimental", multi_input=False, sort_jets=False, particle="b_lep_cart")

# 


def WLEP_model():
    """The Bidirectional LSTM model we use in this study"""
    config = {'act1': 'relu', 'act2': 'relu', 'act3': 'elu',
              'act4': 'relu', 'size1': 400, 'size2': 40, 'size3':40, 'size4': 300,
              'size5': 90, 'size6': 30}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,5), input_shape=(28,)))
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
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    return model

def bHAD_model():
    """The Bidirectional LSTM model we use in this study"""
    config = {'act1': 'relu', 'act2': 'relu', 'act3': 'elu',
              'act4': 'relu', 'size1': 400, 'size2': 40, 'size3':40, 'size4': 300,
              'size5': 90, 'size6': 30}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(7,4), input_shape=(24,)))
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
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    return model




if __name__ == "__main__":
    # Make first model
    model = BDLSTM_model()
    print(model.summary())

    # Load data for first model
