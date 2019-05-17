"""Use this script to load in the model qualititavely evaluating the predictions
made by the model contrasting the predicted value from the actual value"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from models import *
from FormatInputOutput import get_input_output
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# IMPORT MODEL AND LOAD WEIGHTS
model_dir = training_dir
checkpoint_path = "{}/cp.ckpt".format(model_dir)
#model = create_simple_model()
#model = create_regularized_model()
#model = create_model3()
model = create_model4()
print(checkpoint_path)
weights = model.load_weights(checkpoint_path)

###############################################################################
# PICK POINTS TO TEST ON
(training_input, training_output), (testing_input, testing_output) = get_input_output()
indices = np.arange(0, testing_input.shape[0], 1)
np.random.shuffle(indices)

###############################################################################
# USE MODEL TO PREDICT ON FIVE RANDOM TRIALS
predictions = model.predict(testing_input)
for i in range(5):
    print("{}-th trial point".format(indices[i]))
    print("W_had:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[indices[i]][0],
    testing_output[indices[i]][0]))
    print("W_lep:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[indices[i]][1],
            testing_output[indices[i]][1]))
    print("b_had:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[indices[i]][2],
            testing_output[indices[i]][2]))
    print("b_lep:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[indices[i]][3],
            testing_output[indices[i]][3]))
    print("t_had:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[indices[i]][4],
            testing_output[indices[i]][4]))
    print("t_lep:")
    print("Predicted: {0}\nActual: {1}\n".format(predictions[indices[i]][5],
            testing_output[indices[i]][5]))
    print("==================================================================")

###############################################################################
# MAKE PDF FILE COMPARING DIFFERENCES BETWEEN PREDICTIONS AND REAL VALUES
pp = PdfPages('{0}/predictions{1}.pdf'.format(model_dir, model_dir.split("_")[-1]))
xaxis = np.arange(0, testing_input.shape[0], 1)

titles = [
["W_had_Px", "W_had_Py", "W_had_Pz", "W_had_E"],
["W_lep_Px", "W_lep_Py", "W_lep_Pz", "W_lep_E"],
["b_had_Px", "b_had_Py", "b_had_Pz", "b_had_E"],
["b_lep_Px", "b_lep_Py", "b_lep_Pz", "b_lep_E"],
["t_had_Px", "t_had_Py", "t_had_Pz", "t_had_E"],
["t_lep_Px", "t_lep_Py", "t_lep_Pz", "t_lep_E"]
]

for i in range(4):
    fig, sub = plt.subplots(6, 1, figsize=(8, 18), sharex=True)
    for j in range(6):
            sub[j].scatter(xaxis[:10], testing_output[:10,j,i], color='red', label="True")
            sub[j].scatter(xaxis[:10], predictions[:10,j,i], color='blue', label="Predicted")
            sub[j].legend()
            sub[j].set_title(titles[j][i])
            sub[j].set_xlabel("Point Number")
            sub[j].set_ylabel("Value [arb]")
    pp.savefig()
pp.close()
