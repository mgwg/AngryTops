import os, sys
import numpy as np
from ROOT import *
import pickle
import argparse
from array import array
import sklearn.preprocessing
from numpy.linalg import norm
from scipy.spatial import distance
from AngryTops.features import *
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists, undo_scaling

################################################################################
# CONSTANTS
training_dir = sys.argv[1]
representation = sys.argv[2]
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

ALL = 0
NONE = 1
ONLY = 2
b_tagging = NONE    # 0/All: Consider all jets, both b-tagged and not b-tagged
                    # 1/None: Do not consider any b-tagged jets.
                    # 2/Only: Consider only b-tagged jets

# Cut ranges for the partons
W_had_m_cutoff = (30, 130)
W_had_pT_cutoff = (-100, 100)
W_had_dist_cutoff = (0, 0.8)

W_lep_ET_cutoff = (-100, 120)
W_lep_dist_cutoff = (0, 1.0)

b_had_pT_cutoff = (-80, 100)
b_had_dist_cutoff = (0, 0.8)

b_lep_pT_cutoff = (-80, 100)
b_lep_dist_cutoff = (0, 0.8)

np.set_printoptions(precision=3, suppress=True, linewidth=250)
model_filename  = "{}/simple_model.h5".format(training_dir)

################################################################################
# load data
print("INFO: fitting ttbar decay chain...")
predictions = np.load(training_dir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']
event_info = predictions['events']

print("jets shape", jets.shape)
print("b tagging option", b_tagging)
if scaling:
    scaler_filename = training_dir + "scalers.pkl"
    with open( scaler_filename, "rb" ) as file_scaler:
        jets_scalar = pickle.load(file_scaler)
        lep_scalar = pickle.load(file_scaler)
        output_scalar = pickle.load(file_scaler)
    jets_jets, jets_lep, true, fitted = undo_scaling(jets_scalar, lep_scalar, output_scalar, jets, true, fitted)

if not scaling:
    jets_lep = jets[:,:6]
    jets_jets = jets[:,6:]
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6))
    jets_jets = np.delete(jets_jets, 5, 2)

# jets
jet_mu = jets_lep
# First jet for every event
jet_1 = jets_jets[:,0]
# Second jet for every event
jet_2 = jets_jets[:,1]
jet_3 = jets_jets[:,2]
jet_4 = jets_jets[:,3]
jet_5 = jets_jets[:,4]
# Create an array with each jet's arrays for accessing b-tagging states later.
jet_list = np.stack([jet_1, jet_2, jet_3, jet_4, jet_5]) 

# truth
y_true_W_had = true[:,0,:]
y_true_W_lep = true[:,1,:]
y_true_b_had = true[:,2,:]
y_true_b_lep = true[:,3,:]
y_true_t_had = true[:,4,:]
y_true_t_lep = true[:,5,:]

# fitted
y_fitted_W_had = fitted[:,0,:]
y_fitted_W_lep = fitted[:,1,:]
y_fitted_b_had = fitted[:,2,:]
y_fitted_b_lep = fitted[:,3,:]
y_fitted_t_had = fitted[:,4,:]
y_fitted_t_lep = fitted[:,5,:]

# store number of events as a separate variable for clarity
n_events = true.shape[0]
w = 1
print("INFO ...done")

################################################################################
# SET COUNTERS

had_b_btag_total = 0
lep_b_btag_total = 0

good_b_had_total = 0
good_b_lep_total = 0

good_event = 0

################################################################################
# POPULATE TREE
print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
    b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b , representation)

    b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
    b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b, representation)

    jet_mu_vect = MakeP4(jet_mu[i],jet_mu[i][4], representation)

    jet_1_vect = MakeP4(jet_1[i], jet_1[i][4], representation)
    jet_2_vect = MakeP4(jet_2[i], jet_2[i][4], representation)
    jet_3_vect = MakeP4(jet_3[i], jet_3[i][4], representation)
    jet_4_vect = MakeP4(jet_4[i], jet_4[i][4], representation)
    jet_5_vect = MakeP4(jet_5[i], jet_5[i][4], representation)
    
    jets = []
    # add list containing jets of correspoonding event
    jets.append(jet_1_vect)
    jets.append(jet_2_vect)
    jets.append(jet_3_vect)
    jets.append(jet_4_vect)
    # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
    if not np.all(jet_5[i] == 0.):
        jets.append(jet_5_vect)

    ################################################# match jets ################################################# 

    # Set initial distances to be large since we don't know what the minimum distance is yet 
    b_had_dist_true = b_lep_dist_true = 1000000

    # Perform jet matching for the bs, all jets, b-tagged and not b-tagged should be considered.
    for k in range(len(jets)): # loop through each jet to find the minimum distance for each particle
        # For bs:
        b_had_d_true = find_dist(b_had_true, jets[k])
        if b_had_d_true < b_had_dist_true:
            b_had_dist_true = b_had_d_true
            closest_b_had = jets[k]
            b_had_btag = jet_list[k, i, 5]
        b_lep_d_true = find_dist(b_lep_true, jets[k])
        if b_lep_d_true < b_lep_dist_true:
            b_lep_dist_true = b_lep_d_true
            closest_b_lep = jets[k]
            b_lep_btag = jet_list[k, i, 5]
    
    # b quark calculations
    b_had_true_obs_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
    b_lep_true_obs_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()


    ############################################## check whether each event passes cuts #################################################

    # counter for hadronic b
    b_had_pT_cut = (b_had_true_obs_pT_diff >= b_had_pT_cutoff[0] and b_had_true_obs_pT_diff <= b_had_pT_cutoff[1])
    b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
    good_b_had = (b_had_pT_cut and b_had_dist_cut)

    # counter for leptonic b
    b_lep_pT_cut = (b_lep_true_obs_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_obs_pT_diff <= b_lep_pT_cutoff[1])
    b_lep_dist_cut = (b_lep_dist_true <= b_lep_dist_cutoff[1]) 
    good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

    # Good events must pass cuts on all partons.
    good_event += (good_b_had and good_b_lep)

    had_b_btag_total += (good_b_had and b_had_btag)
    lep_b_btag_total += (good_b_lep and b_lep_btag) 
    good_b_had_total += good_b_had
    good_b_lep_total += good_b_lep

print("good hadronic b matches: {}".format(good_b_had_total))
print("good leptonic b matches: {}".format(good_b_lep_total))
print("b-tagged good hadronic b: {}, {}%".format(had_b_btag_total, float(had_b_btag_total)/good_b_had_total*100.0))
print("b-tagged good leptonic b: {}, {}%".format(lep_b_btag_total, float(lep_b_btag_total)/good_b_lep_total*100.0))