# Train the simplest model. Mostly Use this script for testing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from features import *
import os
from models import create_simple_model
from plotting_helper import plot_history
from FormatInputOutput import get_input_output


###############################################################################
# CONSTANTS
BATCH_SIZE = 32
EPOCHES = 5
checkpoint_path = "CheckPoints/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

###############################################################################
# LOADING / PRE-PROCESSING DATA
(training_input, training_output), (testing_input, testing_output) = get_input_output()
print(training_input.shape)

###############################################################################
# BUILDING / TRAINING MODEL
model = create_simple_model()
#model.load_weights(checkpoint_path)
print(model.summary())

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True, verbose=1)

history = model.fit(training_input, training_output,  epochs=EPOCHES,
                    batch_size=BATCH_SIZE, validation_split=0.1,
                    callbacks = [cp_callback]
                    )
###############################################################################
# EVALUATING MODEL
plot_history(history)
test_loss, test_acc = model.evaluate(test_images, test_labels)


###############################################################################
# SAVING AND CLOSING PROGRAM

model.save('CheckPoints/training_1/simple_model.h5')
