import os, sys

date = sys.argv[1]
epochs = sys.argv[2]

from ModelTraining import train_simple_model
train_simple_model.train_model("BDLSTM_model", date, "Feb9.csv", scaling='minmax', rep='pxpypzEM', epochs, sort_jets=False, load_model=False, log_training=True)