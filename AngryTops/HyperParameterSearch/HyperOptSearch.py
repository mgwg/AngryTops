"""Minimize Loss Using MongoDB"""
import numpy as np
from AngryTops.ModelTraining.FormatInputOutput import get_input_output
from hyperopt import fmin, tpe, hp
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow import keras
import tensorflow as tf
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.keras import TuneReporterCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.logger import DEFAULT_LOGGERS


def objective(config, reporter):
    """
    Trains a DNN model for 10 epoches. Return the loss.
    """
    # LOADING / PRE-PROCESSING DATA
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
    = get_input_output(input_filename='topreco_5dec.csv',
                        rep='pxpypz', multi_input=True, scaling='standard')
    # BUILDING / TRAINING MODEL
    model = test_model(config)
    reporter_callback = TuneReporterCallback(reporter)
    history = model.fit(training_input, training_output,  epochs=1,
                            batch_size=32, validation_split=0.1,callbacks=[reporter_callback])


def test_model(config):
    """
    Args:
        config (dict): Parameters provided from the search algorithm
            or variant generation.
        reporter (Reporter): Handle to report intermediate metrics to Tune.
    """
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


if __name__ == "__main__":
    tune.register_trainable('objective', objective)
    ray.init(num_cpus=32, num_gpus=0)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mse",
        mode="min",
        max_t=100,
        grace_period=20)
    space = {
    'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
    'size1': hp.quniform('size1', 1, 200, 1),
    'size2': hp.quniform('size2', 1, 200, 1),
    'size3': hp.quniform('size3', 1, 200, 1),
    'size4': hp.quniform('size4', 1, 200, 1),
    'size5': hp.quniform('size5', 1, 200, 1),
    'size6': hp.quniform('size6', 1, 200, 1),
    'size7': hp.quniform('size7', 1, 200, 1),
    'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
    'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
    'act3': hp.choice('act3', ['relu', 'elu', 'tanh']),
    'act4': hp.choice('act4', ['relu', 'elu', 'tanh']),
    'reg_weight': hp.uniform('reg_weight', 0, 1),
    'rec_weight': hp.uniform('rec_weight', 0, 1)
    }

    algo = HyperOptSearch(space, max_concurrent=8, metric="mse", mode="min")
    results = tune.run(objective, name="my_exp91", num_samples=10, search_alg=algo, resources_per_trial={"cpu": 4, "gpu": 0}, verbose=2, scheduler=sched, loggers=DEFAULT_LOGGERS)
    
