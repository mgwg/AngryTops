"""
Train the simplest model. Mostly Use this script for testing
Meant to be run from the parent directory
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
import pickle
from AngryTops.features import *
from AngryTops.ModelTraining.models import models
from AngryTops.ModelTraining.plotting_helper import plot_history
from AngryTops.ModelTraining.FormatInputOutput import get_input_output
from AngryTops.ModelTraining.custom_loss import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)

print(tf.__version__)
print(tf.test.gpu_device_name())


def train_model(model_name, train_dir, csv_file, log_training=True, load_model=False, **kwargs):
    """
    Trains a DNN model.
    ============================================================================
    INPUTS:
    Model_name: The name of the mode in models.py
    train_dir: Name of the folder to save the training info + model
    csv_file: The csv file to read data from.
    EPOCHES: # of EPOCHES to train
    scaling: Choose between 'standard' or 'minmax' scaling of input and outputs
    rep: The representation of the data. ie. pxpypz vs ptetaphiM vs ...
    multi_input: True if the model is a multi_input model. False otherwise.
    sort_jets: True or False. Sort jets by first btag, then Mass.
    shuffle: If in kwargs.keys, will shuffle the training/testing data.
    weights: The weights for the weighted MSE. Defaults to [1,1,1,1,1,1]
    custom_loss: A custom loss function.
    """
    # CONSTANTS
    if 'retrain' in kwargs.keys():
        train_dir = "PostTraining_Analysis/models/{0}".format(train_dir)
        checkpoint_dir = "{}/checkpoints".format(train_dir)
    else:
        train_dir = "../CheckPoints/{}".format(train_dir)
        checkpoint_dir = "{}/checkpoints".format(train_dir)
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    EPOCHES = kwargs["EPOCHES"]
    BATCH_SIZE = 32
    print("Saving files in: {}".format(train_dir))
    print("Checkpoint Path: ", checkpoint_path)

    ###########################################################################
    # LOGGING
    try:
        log = open("{}/log.txt".format(train_dir), 'w')
    except Exception as e:
        os.mkdir(train_dir)
        log = open("{}/log.txt".format(train_dir), 'w')
    if log_training:
        sys.stdout = log

    ###########################################################################
    # LOADING / PRE-PROCESSING DATA
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
                        = get_input_output(input_filename=csv_file, **kwargs)
    print("Shape of training events: ", training_input.shape)
    print("Shape of testing events: ", testing_input.shape)

    ###########################################################################
    # BUILDING / TRAINING MODEL
    # For the weighted mean square error metric
    if 'weights' in kwargs.keys():
        weights = kwargs['weights']
        print("Weights for Weighted MSE:", weights)
        # Updated the metrics/losses imported from custom_loss.py
        weighted_mse = weighted_MSE(weights)
        custom_metrics["Weighted_MSE"] = weighted_mse
        losses["Weighted_MSE"] = weighted_mse
        metrics.append(weighted_mse)
    if 'custom_loss' in kwargs.keys():
        print("Loss Function: ", kwargs['custom_loss'])
    else:
        print("Loss Function: mse")
    model = models[model_name](metrics, losses, **kwargs)

    # Load previously trained model if it exists
    if load_model:
        try:
            model = tf.keras.models.load_model("%s/simple_model.h5" % train_dir,
                                               custom_objects=custom_metrics)
            print("Loaded weights from previous session")
            print("Loaded weights from previous session", file=sys.stderr)
        except Exception as e:
            print(e)
            print(e, file=sys.stderr)

    print(model.summary())

    # Checkpoint saving / Model training
    filepath = checkpoint_dir + "/weights-improvement-{epoch:02d}.ckpt"
    print("Checkpoints saved in: ", filepath)
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=0)
    try:
        history = model.fit(training_input, training_output,  epochs=EPOCHES,
                            batch_size=BATCH_SIZE, validation_split=0.1,
                            callbacks = [early_stopping, cp_callback]
                            )
    except KeyboardInterrupt:
        print("Training_inerrupted")
        print("Training_inerrupted", file=sys.stderr)
        history = None

    ###########################################################################
    # SAVING MODEL, TRAINING HISTORY AND SCALARS
    model.save('%s/simple_model.h5' % train_dir)
    model.save_weights('%s/model_weights.h5' % train_dir)
    plot_model(model, to_file='%s/model.png' % train_dir)

    scaler_filename = "{}/scalers.pkl".format(train_dir)
    with open( scaler_filename, "wb" ) as file_scaler:
      pickle.dump(jets_scalar, file_scaler, protocol=2)
      pickle.dump(lep_scalar, file_scaler, protocol=2)
      pickle.dump(output_scalar, file_scaler, protocol=2)
    print("INFO: scalers saved to file:", scaler_filename)
    print("INFO: scalers saved to file:", scaler_filename, file=sys.stderr)

    ###########################################################################
    # EVALUATING MODEL
    try:
        test_acc = model.evaluate(testing_input, testing_output)
        print('\nTest accuracy:', test_acc)
        print('\nTest accuracy:', test_acc, file=sys.stderr)
    except Exception as e:
        print(e)

    ###########################################################################
    # MAKE AND SAVE PREDICTIONS
    predictions = model.predict(testing_input)
    if kwargs['multi_input']:
        np.savez("%s/predictions" % train_dir, lep=testing_input[0],
                 jet=testing_input[1], true=testing_output, pred=predictions,
                 events=event_testing)
    else:
        np.savez("{}/predictions".format(train_dir), input=testing_input,
                 true=testing_output, pred=predictions, events=event_testing)

    ###########################################################################
    # SAVE TRAINING HISTORY
    if history is not None:
        for key in history.history.keys():
            np.savez("{0}/{1}.npz".format(train_dir, key),
                            epoches=history.epoch, loss=history.history[key])
        print("Keys: ", history.history.keys())
        plot_history(history, train_dir)

    # for layer in model.layers:
    #     g=layer.get_config()
    #     h=layer.get_weights()
    #     print (g)
    #     print (h)
    sys.stdout = sys.__stdout__
    log.close()

if __name__ == "__main__":
    print("Compiled")
