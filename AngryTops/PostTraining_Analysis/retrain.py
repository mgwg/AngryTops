import numpy as np
from AngryTops.ModelTraining.train_simple_model import train_model
from AngryTops.PostTraining_Analysis.PT_Cutoff import select_pT_events
from AngryTops.ModelTraining.FormatInputOutput import get_input_output
from AngryTops.ModelTraining.models import models

print(tf.__version__)
print(tf.test.gpu_device_name())

# Train Initial Model
print("Training Model for the first iteration")
train_model("stacked_LSTM1", "Retrained_LSTM", "topreco_5dec2.csv",
log_training=False, load_model=False, EPOCHES=100, BATCH_SIZE=32,
scaling='minmax', rep='pxpypzEM', multi_input=False, sort_jets=False,
shuffle=False, training_split=0.5)

# Select events that the model failed to notice the pT cutoff
print("Isolating for events that failed the pT cutoff")
select_pT_events()

# Retrain Model on the selected events
train_model("stacked_LSTM1", "Retrained_LSTM2", "b_had_pT.csv.csv",
log_training=False, load_model=False, EPOCHES=100, BATCH_SIZE=32,
scaling='minmax', rep='pxpypzEM', multi_input=False, sort_jets=False,
shuffle=False, training_split=0.5)
