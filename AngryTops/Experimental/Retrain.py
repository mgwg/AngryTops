import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from AngryTops.features import *
from AngryTops.ModelTraining.single_output_models import *
from AngryTops.ModelTraining.FormatInputOutput import get_input_output
from AngryTops.ModelTraining.models import *
from AngryTops.ModelTraining.custom_loss import *

EPOCHES = 100
BATCH_SIZE=32
(training_input, training_output), (testing_input, testing_output), \
(jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = \
get_input_output(input_filename="topreco_5dec2.csv", scaling='minmax',
rep="experimental", multi_input=False, sort_jets=False, particle="W_lep_cart")

# Make + Train Models
WLEP_model = BDLSTM_model(metrics, losses)
cp_callback = ModelCheckpoint(".", monitor='val_loss', save_weights_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=0)
try:
    history = model.fit(training_input, training_output,  epochs=EPOCHES,
                        batch_size=BATCH_SIZE, validation_split=0.1,
                        callbacks = [early_stopping, cp_callback]
                        )
except Exception as e:
    print("Training Interrupted")

WLEP_model.save('WLep_Model.h5')

# Use the WLEP model to update the training/testing data
(training_input, training_output), (testing_input, testing_output), \
(jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = \
get_input_output(input_filename="topreco_5dec2.csv", scaling='minmax',
rep="experimental", multi_input=False, sort_jets=False, particle="b_lep_cart")

# Add predictions from previous model to the input
pred_train = model.predict(training_input)
pred_test = model.predict(testing_input)
filler_train = np.zeros(shape=(pred_train.shape[0]))
filler_test = np.zeros(shape=(pred_test.shape[0]))
pred_train = np.c_[pred_train, filler_train]
pred_test = np.c_[pred_test, filler_test]
training_input = np.c_[training_input, pred_train]
testing_input = np.c_[testing_input, pred_test]
print(training_input.shape)
print(testing_input.shape)


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
