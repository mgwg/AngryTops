"""Find the ideal hyperparameters for a network architecture"""
from AngryTops.ModelTraining.FormatInputOutput import get_input_output, scale
from hpsklearn import HyperoptEstimator, any_regressor
from hyperopt import tpe
import numpy as np
import sklearn

# Download the data and split into training and test sets
(X_train, y_train), (X_test, y_test), (jets_scalar, lep_scalar, output_scalar), \
(event_training, event_testing) = get_input_output(input_filename='/Users/fardinsyed/Desktop/Top_Quark_Project/AngryTops/csv/topreco_5dec.csv', rep='pxpypzE', scaling=True, multi_input=False, shuffle=True, single_output="target_b_had_Pt")
y_train = y_train.reshape(y_train.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

# Instantiate a HyperoptEstimator with the search space and number of evaluations
estim = HyperoptEstimator(regressor=any_regressor('gradient_boosting_regression'),
                          preprocessing=[],
                          algo=tpe.suggest,
                          max_evals=10,
                          trial_timeout=30000)

# Search the hyperparameter space based on the data
estim.fit(X_train, y_train)

# Show the results

print(estim.score(X_test, y_test))
# 0.962785714286

print(estim.best_model())
