INSTALLATION

NOTE: This README does not contain commands needed to run newer scripts as of August 2021. For the updated files, please refer to the appendices of our report.
======================================================================

Setup the environment

```
virtualenv env-ml
source  env-ml/bin/activate
pip install tensorflow==2.0.0-beta1
pip install sklearn
pip install pandas
pip install ray
```

Install ROOT from src
https://root.cern.ch/downloading-root

Let's assume ROOT is installed under ```$HOME/local/root```

Clone repository

```
git clone https://github.com/IMFardz/AngryTops.git
```

EXECUTE
======================================================================

```
module load cuda/9.0.176
module load cudnn
source $HOME/env-ml/bin/activate
source $HOME/local/root/bin/thisroot.sh
```

PACKAGE DESCRIPTION
======================================================================
AngryTops is a pipeline for training, testing and evaluating neural networks
designed to compute the kinematic reconstruction of ttbar events formed from pp
collisions. This package is divided into five sub-package.

EventGeneration:
Takes ttbar events produced from the Monto Carlo pipeline
(MadGraph8 + Pythia5 + Delphes) and converts them into a csv file format. The
main script, root2csv.py, selects events and specific branches from the ROOT
Trees, applies cuts and writes event info, input columns and output columns into
a csv format. Several helper scripts that are used here are tree_traversal.py,
helper_functions.py and data_augmentation.py. HistCSV.py is used to output the
distributions of the events stored in the csv files.

ModelTraining:
The main pipeline for creating, training and saving neural networks. Different
network architectures are written in models.py and cnn.py. The training and
testing data is withdrawn from the csv files in FormatInputOutput.py. The model
is trained, loaded and saved in train_simple_model.py.

Plotting:
Pipeline for producing the distributions of the true test values and the fitted
values from the network. The scripts are run in the order:
fit.py -> histograms.py -> plot_observables.py. For models that output only a
single distribution, use plot_single.py should be used instead.

HyperParameterSearch:
Pipeline for searching hyper-parameter spaces. Model architectures are defined in
test_models.py and their corresponding hyper-parameter spaces are in
param_spaces.py. HyperOptSearch.py is the main script used for this pipeline.

PostTraining_Analysis:
Routine for training a model and then retraining the model on a subset of
events, determined

OTHER FOLDERS/SCRIPTS
======================================================================
CheckPoints: Where trained models are saved.
archive: Contains old scripts
AtlasDocs: Used for formatting plots with ROOT
ray_results: Contains results from hyperparameter searches
combine_images.py: Used to create comparison plots between models. 
