"""Use this script to load in the model qualititavely evaluating the predictions
made by the model contrasting the predicted value from the actual value"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from models import create_simple_model
from FormatInputOutput import get_input_output

###############################################################################
# IMPORT MODEL AND LOAD WEIGHTS
checkpoint_path = "CheckPoints/training_1/cp.ckpt"
model = create_simple_model()
weights = model.load_weights(checkpoint_path)

###############################################################################
# PICK POINTS TO TEST ON
(training_input, training_output), (testing_input, testing_output) = get_input_output()
trial_tests = testing_input[-5:]
trial_targets = testing_output[-5:]
print(trial_targets)

###############################################################################
# USE MODEL TO PREDICT ON THE TRIALS
predictions = model.predict(trial_tests)
for i in range(len(trial_tests)):
    print("{}-th trial point")
    print("W_had:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[i][0], trial_targets[i][0]))
    print("W_lep:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[i][1], trial_targets[i][1]))
    print("b_had:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[i][2], trial_targets[i][2]))
    print("b_lep:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[i][3], trial_targets[i][3]))
    print("t_had:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[i][4], trial_targets[i][4]))
    print("t_lep:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[i][5], trial_targets[i][5]))
