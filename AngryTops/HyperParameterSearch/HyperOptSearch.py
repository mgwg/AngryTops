"""Minimize Loss Using MongoDB"""
import numpy as np
import sys
from hyperopt import hp
from AngryTops.ModelTraining.FormatInputOutput import get_input_output
from AngryTops.HyperParameterSearch.test_models import test_models
from AngryTops.HyperParameterSearch.param_spaces import parameter_spaces
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.keras import TuneReporterCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.logger import DEFAULT_LOGGERS

# The corresponding model and space for testing
model_name = sys.argv[1]
space_name = sys.argv[2]
search_name = sys.argv[3]

rep = 'pxpypz'
scaling = 'standard'
multi_input = True
if len(sys.argv) > 4:
    rep = sys.argv[4]
    scaling = sys.argv[5]
if len(sys.argv) > 6: multi_input = False
test_model = test_models[model_name]
space = parameter_spaces[space_name]

def objective(config, reporter, **kwargs):
    """
    Trains a DNN model for 10 epoches. Return the loss.
    """
    # LOADING / PRE-PROCESSING DATA
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
    = get_input_output(input_filename='topreco_5dec.csv',
                        rep=rep, multi_input=multi_input, scaling=scaling)
    # BUILDING / TRAINING MODEL
    model = test_model(config)
    reporter_callback = TuneReporterCallback(reporter)
    history = model.fit(training_input, training_output,  epochs=1,
              batch_size=32, validation_split=0.1,callbacks=[reporter_callback])


if __name__ == "__main__":
    tune.register_trainable('objective', objective)
    ray.init(num_cpus=32, num_gpus=0)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mse",
        mode="min",
        max_t=10000,
        grace_period=20)

    algo = HyperOptSearch(space, max_concurrent=8, metric="mse", mode="min")
    results = tune.run(objective, name=search_name, num_samples=1000,
                       search_alg=algo, resources_per_trial={"cpu": 4, "gpu": 0},
                       verbose=2, scheduler=sched, loggers=DEFAULT_LOGGERS)
