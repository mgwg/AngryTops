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

###############################################################################
# CONSTANTS
BATCH_SIZE = 32
EPOCHES = 30
if len(sys.argv) > 1:
    training_dir = "CheckPoints/{}".format(sys.argv[1])
    print("Saving files in: {}".format(training_dir))
    checkpoint_path = "{}/{}/cp.ckpt".format("CheckPoints", sys.argv[1])
    model_num = int(sys.argv[2])
else:
    checkpoint_path = "{}/cp.ckpt".format(training_dir)
    model_num = 0

###############################################################################
# LOADING / PRE-PROCESSING DATA
(training_input, training_output), (testing_input, testing_output), \
       (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = get_input_output()
testing_input_original = testing_input.copy()
print(training_input.shape)


###############################################################################
# BUILDING / TRAINING MODEL
model = models[model_num]()
try:
    model.load_weights(checkpoint_path)
    print("Loaded weights from previous training session")
except Exception as e:
    print(e)
#model = keras.models.load_model("{}/simple_model.h5".format(training_dir))
print(model.summary())

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True, verbose=1)
try:
    history = model.fit(training_input, training_output,  epochs=EPOCHES,
                        batch_size=BATCH_SIZE, validation_split=0.1,
                        callbacks = [cp_callback]
                        )
except ValueError:
    print("Detected invalid input shape. Assuming model is multi-input")
    training_input = [training_input[:,:6], training_input[:,6:]]
    testing_input = [testing_input[:,:6], testing_input[:,6:]]
    history = model.fit(training_input, training_output,  epochs=EPOCHES,
                        batch_size=BATCH_SIZE, validation_split=0.1,
                        callbacks = [cp_callback]
                        )
except KeyboardInterrupt:
    print("Training_inerrupted")
    history = None


###############################################################################
# SAVING MODEL, TRAINING HISTORY AND SCALARS
model.save('{}/simple_model.h5'.format(training_dir))

scaler_filename = "{}/scalers.pkl".format(training_dir)
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump(jets_scalar, file_scaler, protocol=2)
  pickle.dump(lep_scalar, file_scaler, protocol=2)
  pickle.dump(output_scalar, file_scaler, protocol=2)
print("INFO: scalers saved to file:", scaler_filename)

###############################################################################
# EVALUATING MODEL AND MAKE PREDICTIONS
if history is not None:
    for key in history.history.keys():
        np.savez("{0}/{1}.npz".format(training_dir, key), epoches=history.epoch, loss=history.history[key])
    print(history.history.keys())
    plot_history(history, training_dir)

# Evaluating model and saving the predictions
test_acc = model.evaluate(testing_input, testing_output)
print('\nTest accuracy:', test_acc)
predictions = model.predict(testing_input)
np.savez("{}/predictions".format(training_dir), input=testing_input_original,\
         true=testing_output, pred=predictions, events=event_testing)
