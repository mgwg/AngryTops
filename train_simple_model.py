# Train the simplest model. Mostly Use this script for testing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from features import *
import os
from models import *
from plotting_helper import plot_history
from FormatInputOutput import get_input_output
import pickle

###############################################################################
# CONSTANTS
BATCH_SIZE = 32
EPOCHES = 1
checkpoint_path = "{}/cp.ckpt".format(training_dir)

###############################################################################
# LOADING / PRE-PROCESSING DATA
(training_input, training_output), (testing_input, testing_output), \
       (jets_scalar, lep_scalar, output_scalar) = get_input_output()
print(training_input.shape)


###############################################################################
# BUILDING / TRAINING MODEL
model = create_model5()
try:
    model.load_weights(checkpoint_path)
    print("Loaded weights from previous training session")
except Exception as e:
    print(e)
#model = keras.models.load_model("{}/simple_model.h5".format(training_dir))
print(model.summary())

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True, verbose=1)

history = model.fit(training_input, training_output,  epochs=EPOCHES,
                    batch_size=BATCH_SIZE, validation_split=0.1,
                    callbacks = [cp_callback]
                    )

###############################################################################
# SAVING MODEL, TRAINING HISTORY AND SCALARS
model.save('{}/simple_model.h5'.format(training_dir))
for key in history.history.keys():
    np.savez("{0}/{1}.npz".format(training_dir, key), epoches=history.epoch, loss=history.history[key])

scaler_filename = "{}/scalers.pkl".format(training_dir)
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump(jets_scalar, file_scaler)
  pickle.dump(lep_scalar, file_scaler)
  pickle.dump(output_scalar, file_scaler)
print("INFO: scalers saved to file:", scaler_filename)

###############################################################################
# EVALUATING MODEL AND MAKE PREDICTIONS
print(history.history.keys())
plot_history(history, training_dir)
test_acc = model.evaluate(testing_input, testing_output)
print('\nTest accuracy:', test_acc)
predictions = model.predict(testing_input)
np.savez("{}/predictions".format(training_dir), input=testing_input, true=testing_output, pred=predictions)
