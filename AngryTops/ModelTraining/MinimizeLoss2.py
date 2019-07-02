import numpy as np
from AngryTops.ModelTraining.FormatInputOutput import get_input_output
from hyperopt import fmin, tpe, hp
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow import keras
import tensorflow as tf


def objective2(args): return objective(*args)

def objective(learn_rate, size1, size2, size3, size4, size5, size6, size7,\
               size8, size9, size10, act1, act2, act3, act4, act5, act6, act7, \
               reg_weight, rec_weight):
    """
    Trains a DNN model for 10 epoches. Return the loss.
    """
    ###########################################################################
    # LOADING / PRE-PROCESSING DATA
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
    = get_input_output(input_filename='topreco_5dec.csv',
                        rep='pxpypz', multi_input=True, scaling='standard')

    ###########################################################################
    # BUILDING / TRAINING MODEL
    model = test_model(learn_rate, size1, size2, size3, size4, size5, size6, size7,\
                   act1, act2, act3, act4, reg_weight, rec_weight)
    try:
        history = model.fit(training_input, training_output,  epochs=1,
                            batch_size=32, validation_split=0.1,)
    except KeyboardInterrupt:
        print("Training_inerrupted")
        history = None

    ###########################################################################
    # EVALUATING MODEL AND MAKE PREDICTIONS
    # Evaluating model and saving the predictions
    test_acc = model.evaluate(testing_input, testing_output)
    print('\nTest accuracy:', test_acc)
    return test_acc[-1]


def test_model(learn_rate, size1, size2, size3, size4, size5, size6, size7,\
               size8, size9, size10, act1, act2, act3, act4, act5, act6, act7, \
               reg_weight, rec_weight):
    """Froms our model for testing"""
    """A denser version of model_multi"""
    print("INPUT")
    print("learn_rate: {0}\nsize1: {1}\nsize2: {2}\nsize3: {3}\nsize4: {4}\n\
          size5 {5}\nsize6: {6}\nsize7: {7}\nact1: {8}\nact2: {9}\nact3: {10}\n\
          act4: {11}\nreg_weight: {12}\nrec_weight: {13}".format(learn_rate, size1, \
          size2, size3, size4, size5, size6, size7, act1, act2, act3, act4, reg_weight, rec_weight))

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(size1), return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(int(size2), return_sequences=True,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = LSTM(int(size3), return_sequences=False,
                  kernel_regularizer=l2(reg_weight),
                  recurrent_regularizer=l2(rec_weight))(x_jets)
    x_jets = Dense(int(size4), activation=act1)(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(size5), activation=act2)(input_lep)
    x_lep = Dense(int(size6), activation=act3)(input_lep)
    x_lep = Dense(int(size7), activation=act4)(input_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(size8), activation=act5)(combined)
    final = Dense(int(size9), activation=act6)(final)
    final = Dense(int(size10), activation=act7)(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model


if __name__ == "__main__":
    space = [
    hp.uniform('learn_rate', 10e-7, 10e-1),
    hp.quniform('size1', 1, 200, 1),
    hp.quniform('size2', 1, 200, 1),
    hp.quniform('size3', 1, 200, 1),
    hp.quniform('size4', 1, 200, 1),
    hp.quniform('size5', 1, 200, 1),
    hp.quniform('size6', 1, 200, 1),
    hp.quniform('size7', 1, 200, 1),
    hp.quniform('size8', 1, 200, 1),
    hp.quniform('size9', 1, 200, 1),
    hp.quniform('size10', 1, 200, 1),
    hp.choice('act1', ['relu', 'elu', 'tanh']),
    hp.choice('act2', ['relu', 'elu', 'tanh']),
    hp.choice('act3', ['relu', 'elu', 'tanh']),
    hp.choice('act4', ['relu', 'elu', 'tanh']),
    hp.choice('act5', ['relu', 'elu', 'tanh']),
    hp.choice('act6', ['relu', 'elu', 'tanh']),
    hp.choice('act7', ['relu', 'elu', 'tanh']),
    hp.uniform('reg_weight', 0, 1),
    hp.uniform('rec_weight', 0, 1)
    ]

    best = fmin(fn=objective2,
    space=space,
    algo=tpe.suggest,
    max_evals=300)
    print(best)
