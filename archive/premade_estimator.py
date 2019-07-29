import tensorflow as tf
from AngryTops.ModelTraining.FormatInputOutput import *

(training_input, training_output), (testing_input, testing_output), \
(jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
    = get_input_output(input_filename="topreco_5dec2.csv", scaling='minmax',
                       rep="pxpypzEM", multi_input=True, sort_jets=False)
