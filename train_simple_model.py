# Train the simplest model. Mostly Use this script for testing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from features import *
import os
import sys
from models import models
from plotting_helper import plot_history
from FormatInputOutput import get_input_output
import pickle

print(tf.__version__)
print(tf.test.gpu_device_name())

def train_model(model_num, csv_file="csv/topreco.csv", BATCH_SIZE=32, EPOCHES=30,\
                    train_dir=training_dir, learn_rate=0.001, scaling="minmax",\
                    rep="cart", input_size=30, reshape_shape=(6,6), **kwargs):
###############################################################################
    # CONSTANTS
    train_dir = "CheckPoints/{}".format(train_dir)
    print("Saving files in: {}".format(train_dir))
    checkpoint_path = "{}/cp.ckpt".format(train_dir)
    try:
        log = open("{}/log.txt".format(train_dir), 'w')
        sys.stdout = log
    except Exception:
        os.mkdir(train_dir)
        log = open("{}/log.txt".format(train_dir), 'w')
        sys.stdout = log

###############################################################################
    # LOADING / PRE-PROCESSING DATA
    (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), \
           (event_training, event_testing) = get_input_output(type="minmax", rep=rep)
    testing_input_original = testing_input.copy()
    print("Shape of training_input: {}".format(training_input.shape))
    print("Shape of training_input: {}".format(training_input.shape), file=sys.stderr)
    print("Shape of testing_input: {}".format(testing_input.shape), file=sys.stderr)


###############################################################################
    # BUILDING / TRAINING MODEL
    model = models[model_num](learn_rate, **kwargs)
    try:
        model.load_weights(checkpoint_path)
        print("Loaded weights from previous training session")
        print("Loaded weights from previous training session", file=sys.stderr)
    except Exception as e:
        print(e)
        print(e, file=sys.stderr)
    #model = keras.models.load_model("{}/simple_model.h5".format(train_dir))
    print(model.summary())
    print(model.summary(), file=sys.stderr)

    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True, verbose=1)
    try:
        history = model.fit(training_input, training_output,  epochs=EPOCHES,
                            batch_size=BATCH_SIZE, validation_split=0.1,
                            callbacks = [cp_callback]
                            )
    except ValueError:
        print("Detected invalid input shape. Assuming model is multi-input")
        print("Detected invalid input shape. Assuming model is multi-input", file=sys.stderr)
        try:
            training_input = [training_input[:,:6], training_input[:,6:]]
            testing_input = [testing_input[:,:6], testing_input[:,6:]]
            history = model.fit(training_input, training_output,  epochs=EPOCHES,
                            batch_size=BATCH_SIZE, validation_split=0.1,
                            callbacks = [cp_callback]
                            )
        except KeyboardInterrupt:
            print("Training_inerrupted")
            print("Training_inerrupted", file=sys.stderr)
            history = None
    except KeyboardInterrupt:
        print("Training_inerrupted")
        print("Training_inerrupted", file=sys.stderr)
        history = None

###############################################################################
    # SAVING MODEL, TRAINING HISTORY AND SCALARS
    model.save('{}/simple_model.h5'.format(train_dir))

    scaler_filename = "{}/scalers.pkl".format(train_dir)
    with open( scaler_filename, "wb" ) as file_scaler:
      pickle.dump(jets_scalar, file_scaler, protocol=2)
      pickle.dump(lep_scalar, file_scaler, protocol=2)
      pickle.dump(output_scalar, file_scaler, protocol=2)
    print("INFO: scalers saved to file:", scaler_filename)
    print("INFO: scalers saved to file:", scaler_filename, file=sys.stderr)

###############################################################################
    # EVALUATING MODEL AND MAKE PREDICTIONS
    # Evaluating model and saving the predictions
    test_acc = model.evaluate(testing_input, testing_output)
    print('\nTest accuracy:', test_acc)
    print('\nTest accuracy:', test_acc, file=sys.stderr)
    predictions = model.predict(testing_input)
    np.savez("{}/predictions".format(train_dir), input=testing_input_original,\
             true=testing_output, pred=predictions, events=event_testing)

    if history is not None:
        for key in history.history.keys():
            np.savez("{0}/{1}.npz".format(train_dir, key), epoches=history.epoch, loss=history.history[key])
        print("Keys: ", history.history.keys())
        plot_history(history, train_dir)

    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        train_model(int(sys.argv[1]), train_dir=sys.argv[2])
    elif len(sys.argv) == 4:
        train_model(int(sys.argv[1]), train_dir=sys.argv[2], csv_file=sys.argv[3])
    elif len(sys.argv) == 5:
        train_model(int(sys.argv[1]), train_dir=sys.argv[2], csv_file=sys.argv[3],
                    learn_rate = np.float(sys.argv[4]))
    elif len(sys.argv) == 6:
        train_model(int(sys.argv[1]), train_dir=sys.argv[2], csv_file=sys.argv[3],
                    learn_rate = np.float(sys.argv[4]), scaling=sys.argv[5])
