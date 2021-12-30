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
event_type = sys.argv[3]    # good (reconstructable), bad (unreconstructable), or all
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

print("creating tree for {} events".format(event_type))

################################################################################
# load data
print("INFO: fitting ttbar decay chain...")
predictions = np.load(training_dir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']
event_info = predictions['events']

particles_shape = (true.shape[1], true.shape[2])
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
# CREATE OUTPUT TREE/FILE
ofilename = "{}/tree_fitted_{}.root".format(training_dir, event_type)
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

# Create output tree
b_eventNumber = array( 'l', [ 0 ] )
b_runNumber   = array( 'i', [ 0 ] )
b_mcChannelNumber = array( 'i', [ 0 ] )
b_weight_mc   = array( 'f', [ 0.] )

b_W_had_px_obs   = array( 'f', [ -1.] )
b_W_had_py_obs   = array( 'f', [ -1.] )
b_W_had_pz_obs   = array( 'f', [ -1.] )
b_W_had_E_obs    = array( 'f', [ -1.] )
b_W_had_m_obs    = array( 'f', [ -1.] )
b_W_had_pt_obs   = array( 'f', [ -1.] )
b_W_had_y_obs    = array( 'f', [ -1.] )
b_W_had_phi_obs  = array( 'f', [ -1.] )
b_W_had_num_jets  = array( 'f', [ -1.] )
b_b_had_px_obs   = array( 'f', [ -1.] )
b_b_had_py_obs   = array( 'f', [ -1.] )
b_b_had_pz_obs   = array( 'f', [ -1.] )
b_b_had_E_obs    = array( 'f', [ -1.] )
b_b_had_m_obs    = array( 'f', [ -1.] )
b_b_had_pt_obs   = array( 'f', [ -1.] )
b_b_had_y_obs    = array( 'f', [ -1.] )
b_b_had_phi_obs  = array( 'f', [ -1.] )

b_t_had_px_obs   = array( 'f', [ -1.] )
b_t_had_py_obs   = array( 'f', [ -1.] )
b_t_had_pz_obs   = array( 'f', [ -1.] )
b_t_had_E_obs    = array( 'f', [ -1.] )
b_t_had_m_obs    = array( 'f', [ -1.] )
b_t_had_pt_obs   = array( 'f', [ -1.] )
b_t_had_y_obs    = array( 'f', [ -1.] )
b_t_had_phi_obs  = array( 'f', [ -1.] )

b_W_lep_px_obs   = array( 'f', [ -1.] )
b_W_lep_py_obs   = array( 'f', [ -1.] )
b_W_lep_Et_obs    = array( 'f', [ -1.] )
b_W_lep_phi_obs  = array( 'f', [ -1.] )
b_b_lep_px_obs   = array( 'f', [ -1.] )
b_b_lep_py_obs   = array( 'f', [ -1.] )
b_b_lep_pz_obs   = array( 'f', [ -1.] )
b_b_lep_E_obs    = array( 'f', [ -1.] )
b_b_lep_m_obs    = array( 'f', [ -1.] )
b_b_lep_pt_obs   = array( 'f', [ -1.] )
b_b_lep_y_obs    = array( 'f', [ -1.] )
b_b_lep_phi_obs  = array( 'f', [ -1.] )

b_t_lep_px_obs   = array( 'f', [ -1.] )
b_t_lep_py_obs   = array( 'f', [ -1.] )
b_t_lep_pt_obs   = array( 'f', [ -1.] )
b_t_lep_phi_obs  = array( 'f', [ -1.] )

b_jet1_px_obs   = array( 'f', [ -1.] )
b_jet1_py_obs   = array( 'f', [ -1.] )
b_jet1_pz_obs   = array( 'f', [ -1.] )
b_jet1_pt_obs   = array( 'f', [ -1.] )
b_jet1_E_obs    = array( 'f', [ -1.] )
b_jet1_m_obs    = array( 'f', [ -1.] )
b_jet1_btag_obs   = array( 'f', [ -1.] )
b_jet2_px_obs   = array( 'f', [ -1.] )
b_jet2_py_obs   = array( 'f', [ -1.] )
b_jet2_pz_obs   = array( 'f', [ -1.] )
b_jet2_pt_obs   = array( 'f', [ -1.] )
b_jet2_E_obs    = array( 'f', [ -1.] )
b_jet2_m_obs    = array( 'f', [ -1.] )
b_jet2_btag_obs   = array( 'f', [ -1.] )
b_jet3_px_obs   = array( 'f', [ -1.] )
b_jet3_py_obs   = array( 'f', [ -1.] )
b_jet3_pz_obs   = array( 'f', [ -1.] )
b_jet3_pt_obs   = array( 'f', [ -1.] )
b_jet3_E_obs    = array( 'f', [ -1.] )
b_jet3_m_obs    = array( 'f', [ -1.] )
b_jet3_btag_obs   = array( 'f', [ -1.] )
b_jet4_px_obs   = array( 'f', [ -1.] )
b_jet4_py_obs   = array( 'f', [ -1.] )
b_jet4_pz_obs   = array( 'f', [ -1.] )
b_jet4_pt_obs   = array( 'f', [ -1.] )
b_jet4_E_obs    = array( 'f', [ -1.] )
b_jet4_m_obs    = array( 'f', [ -1.] )
b_jet4_btag_obs   = array( 'f', [ -1.] )
b_jet5_px_obs   = array( 'f', [ -1.] )
b_jet5_py_obs   = array( 'f', [ -1.] )
b_jet5_pz_obs   = array( 'f', [ -1.] )
b_jet5_pt_obs   = array( 'f', [ -1.] )
b_jet5_E_obs    = array( 'f', [ -1.] )
b_jet5_m_obs    = array( 'f', [ -1.] )
b_jet5_btag_obs   = array( 'f', [ -1.] )
b_jetmu_px_obs   = array( 'f', [ -1.] )
b_jetmu_py_obs   = array( 'f', [ -1.] )
b_jetmu_pz_obs   = array( 'f', [ -1.] )
b_jetmu_T0_obs    = array( 'f', [ -1.] )
b_jetlep_ET_obs    = array( 'f', [ -1.] )
b_jetlep_phi_obs   = array( 'f', [ -1.] )

b_W_had_px_true   = array( 'f', [ -1.] )
b_W_had_py_true   = array( 'f', [ -1.] )
b_W_had_pz_true   = array( 'f', [ -1.] )
b_W_had_E_true    = array( 'f', [ -1.] )
b_W_had_m_true    = array( 'f', [ -1.] )
b_W_had_pt_true   = array( 'f', [ -1.] )
b_W_had_y_true    = array( 'f', [ -1.] )
b_W_had_phi_true  = array( 'f', [ -1.] )
b_b_had_px_true   = array( 'f', [ -1.] )
b_b_had_py_true   = array( 'f', [ -1.] )
b_b_had_pz_true   = array( 'f', [ -1.] )
b_b_had_E_true    = array( 'f', [ -1.] )
b_b_had_m_true    = array( 'f', [ -1.] )
b_b_had_pt_true   = array( 'f', [ -1.] )
b_b_had_y_true    = array( 'f', [ -1.] )
b_b_had_phi_true  = array( 'f', [ -1.] )
b_t_had_px_true   = array( 'f', [ -1.] )
b_t_had_py_true   = array( 'f', [ -1.] )
b_t_had_pz_true   = array( 'f', [ -1.] )
b_t_had_E_true    = array( 'f', [ -1.] )
b_t_had_m_true    = array( 'f', [ -1.] )
b_t_had_pt_true   = array( 'f', [ -1.] )
b_t_had_y_true    = array( 'f', [ -1.] )
b_t_had_phi_true  = array( 'f', [ -1.] )
b_W_lep_px_true   = array( 'f', [ -1.] )
b_W_lep_py_true   = array( 'f', [ -1.] )
b_W_lep_pz_true   = array( 'f', [ -1.] )
b_W_lep_E_true    = array( 'f', [ -1.] )
b_W_lep_m_true    = array( 'f', [ -1.] )
b_W_lep_pt_true   = array( 'f', [ -1.] )
b_W_lep_y_true    = array( 'f', [ -1.] )
b_W_lep_phi_true  = array( 'f', [ -1.] )
b_b_lep_px_true   = array( 'f', [ -1.] )
b_b_lep_py_true   = array( 'f', [ -1.] )
b_b_lep_pz_true   = array( 'f', [ -1.] )
b_b_lep_E_true    = array( 'f', [ -1.] )
b_b_lep_m_true    = array( 'f', [ -1.] )
b_b_lep_pt_true   = array( 'f', [ -1.] )
b_b_lep_y_true    = array( 'f', [ -1.] )
b_b_lep_phi_true  = array( 'f', [ -1.] )
b_t_lep_px_true   = array( 'f', [ -1.] )
b_t_lep_py_true   = array( 'f', [ -1.] )
b_t_lep_pz_true   = array( 'f', [ -1.] )
b_t_lep_E_true    = array( 'f', [ -1.] )
b_t_lep_m_true    = array( 'f', [ -1.] )
b_t_lep_pt_true   = array( 'f', [ -1.] )
b_t_lep_y_true    = array( 'f', [ -1.] )
b_t_lep_phi_true  = array( 'f', [ -1.] )

b_W_had_px_fitted   = array( 'f', [ -1.] )
b_W_had_py_fitted   = array( 'f', [ -1.] )
b_W_had_pz_fitted   = array( 'f', [ -1.] )
b_W_had_E_fitted    = array( 'f', [ -1.] )
b_W_had_m_fitted    = array( 'f', [ -1.] )
b_W_had_pt_fitted   = array( 'f', [ -1.] )
b_W_had_y_fitted    = array( 'f', [ -1.] )
b_W_had_phi_fitted  = array( 'f', [ -1.] )
b_b_had_px_fitted   = array( 'f', [ -1.] )
b_b_had_py_fitted   = array( 'f', [ -1.] )
b_b_had_pz_fitted   = array( 'f', [ -1.] )
b_b_had_E_fitted    = array( 'f', [ -1.] )
b_b_had_m_fitted    = array( 'f', [ -1.] )
b_b_had_pt_fitted   = array( 'f', [ -1.] )
b_b_had_y_fitted    = array( 'f', [ -1.] )
b_b_had_phi_fitted  = array( 'f', [ -1.] )
b_t_had_px_fitted   = array( 'f', [ -1.] )
b_t_had_py_fitted   = array( 'f', [ -1.] )
b_t_had_pz_fitted   = array( 'f', [ -1.] )
b_t_had_E_fitted    = array( 'f', [ -1.] )
b_t_had_m_fitted    = array( 'f', [ -1.] )
b_t_had_pt_fitted   = array( 'f', [ -1.] )
b_t_had_y_fitted    = array( 'f', [ -1.] )
b_t_had_phi_fitted  = array( 'f', [ -1.] )
b_W_lep_px_fitted   = array( 'f', [ -1.] )
b_W_lep_py_fitted   = array( 'f', [ -1.] )
b_W_lep_pz_fitted   = array( 'f', [ -1.] )
b_W_lep_E_fitted    = array( 'f', [ -1.] )
b_W_lep_m_fitted    = array( 'f', [ -1.] )
b_W_lep_pt_fitted   = array( 'f', [ -1.] )
b_W_lep_y_fitted    = array( 'f', [ -1.] )
b_W_lep_phi_fitted  = array( 'f', [ -1.] )
b_b_lep_px_fitted   = array( 'f', [ -1.] )
b_b_lep_py_fitted   = array( 'f', [ -1.] )
b_b_lep_pz_fitted   = array( 'f', [ -1.] )
b_b_lep_E_fitted    = array( 'f', [ -1.] )
b_b_lep_m_fitted    = array( 'f', [ -1.] )
b_b_lep_pt_fitted   = array( 'f', [ -1.] )
b_b_lep_y_fitted    = array( 'f', [ -1.] )
b_b_lep_phi_fitted  = array( 'f', [ -1.] )
b_t_lep_px_fitted   = array( 'f', [ -1.] )
b_t_lep_py_fitted   = array( 'f', [ -1.] )
b_t_lep_pz_fitted   = array( 'f', [ -1.] )
b_t_lep_E_fitted    = array( 'f', [ -1.] )
b_t_lep_m_fitted    = array( 'f', [ -1.] )
b_t_lep_pt_fitted   = array( 'f', [ -1.] )
b_t_lep_y_fitted    = array( 'f', [ -1.] )
b_t_lep_phi_fitted  = array( 'f', [ -1.] )

tree = TTree( "nominal", "nominal" )
tree.Branch( "eventNumber",     b_eventNumber,     'eventNumber/l' )
tree.Branch( 'runNumber',       b_runNumber,       'runNumber/i' )
tree.Branch( 'mcChannelNumber', b_mcChannelNumber, 'mcChannelNumber/i' )
tree.Branch( 'weight_mc',       b_weight_mc,       'weight_mc/F' )

tree.Branch( 'W_had_px_obs',   b_W_had_px_obs,   'W_had_px_obs/F' )
tree.Branch( 'W_had_py_obs',   b_W_had_py_obs,   'W_had_py_obs/F' )
tree.Branch( 'W_had_pz_obs',   b_W_had_pz_obs,   'W_had_pz_obs/F' )
tree.Branch( 'W_had_E_obs',    b_W_had_E_obs,    'W_had_E_obs/F' )
tree.Branch( 'W_had_m_obs',    b_W_had_m_obs,    'W_had_m_obs/F' )
tree.Branch( 'W_had_pt_obs',   b_W_had_pt_obs,   'W_had_pt_obs/F' )
tree.Branch( 'W_had_y_obs',    b_W_had_y_obs,    'W_had_y_obs/F' )
tree.Branch( 'W_had_phi_obs',  b_W_had_phi_obs,  'W_had_phi_obs/F' )
tree.Branch( 'W_had_num_jets',  b_W_had_num_jets,  'W_had_num_jets/F' )
tree.Branch( 'b_had_px_obs',   b_b_had_px_obs,   'b_had_px_obs/F' )
tree.Branch( 'b_had_py_obs',   b_b_had_py_obs,   'b_had_py_obs/F' )
tree.Branch( 'b_had_pz_obs',   b_b_had_pz_obs,   'b_had_pz_obs/F' )
tree.Branch( 'b_had_E_obs',    b_b_had_E_obs,    'b_had_E_obs/F' )
tree.Branch( 'b_had_m_obs',    b_b_had_m_obs,    'b_had_m_obs/F' )
tree.Branch( 'b_had_pt_obs',   b_b_had_pt_obs,   'b_had_pt_obs/F' )
tree.Branch( 'b_had_y_obs',    b_b_had_y_obs,    'b_had_y_obs/F' )
tree.Branch( 'b_had_phi_obs',  b_b_had_phi_obs,  'b_had_phi_obs/F' )
tree.Branch( 't_had_px_obs',   b_t_had_px_obs,   't_had_px_obs/F' )
tree.Branch( 't_had_py_obs',   b_t_had_py_obs,   't_had_py_obs/F' )
tree.Branch( 't_had_pz_obs',   b_t_had_pz_obs,   't_had_pz_obs/F' )
tree.Branch( 't_had_E_obs',    b_t_had_E_obs,    't_had_E_obs/F' )
tree.Branch( 't_had_m_obs',    b_t_had_m_obs,    't_had_m_obs/F' )
tree.Branch( 't_had_pt_obs',   b_t_had_pt_obs,   't_had_pt_obs/F' )
tree.Branch( 't_had_y_obs',    b_t_had_y_obs,    't_had_y_obs/F' )
tree.Branch( 't_had_phi_obs',  b_t_had_phi_obs,  't_had_phi_obs/F' )
tree.Branch( 'W_lep_px_obs',   b_W_lep_px_obs,   'W_lep_px_obs/F' )
tree.Branch( 'W_lep_py_obs',   b_W_lep_py_obs,   'W_lep_py_obs/F' )
tree.Branch( 'W_lep_Et_obs',    b_W_lep_Et_obs,    'W_lep_Et_obs/F' )
tree.Branch( 'W_lep_phi_obs',  b_W_lep_phi_obs,  'W_lep_phi_obs/F' )
tree.Branch( 'b_lep_px_obs',   b_b_lep_px_obs,   'b_lep_px_obs/F' )
tree.Branch( 'b_lep_py_obs',   b_b_lep_py_obs,   'b_lep_py_obs/F' )
tree.Branch( 'b_lep_pz_obs',   b_b_lep_pz_obs,   'b_lep_pz_obs/F' )
tree.Branch( 'b_lep_E_obs',    b_b_lep_E_obs,    'b_lep_E_obs/F' )
tree.Branch( 'b_lep_m_obs',    b_b_lep_m_obs,    'b_lep_m_obs/F' )
tree.Branch( 'b_lep_pt_obs',   b_b_lep_pt_obs,   'b_lep_pt_obs/F' )
tree.Branch( 'b_lep_y_obs',    b_b_lep_y_obs,    'b_lep_y_obs/F' )
tree.Branch( 'b_lep_phi_obs',  b_b_lep_phi_obs,  'b_lep_phi_obs/F' )
tree.Branch( 't_lep_px_obs',   b_t_lep_px_obs,   't_lep_px_obs/F' )
tree.Branch( 't_lep_py_obs',   b_t_lep_py_obs,   't_lep_py_obs/F' )
tree.Branch( 't_lep_pt_obs',   b_t_lep_pt_obs,   't_lep_pt_obs/F' )
tree.Branch( 't_lep_phi_obs',  b_t_lep_phi_obs,  't_lep_phi_true/F' )

tree.Branch( 'jet1_px_obs', b_jet1_px_obs, 'jet1_px_obs/F')
tree.Branch( 'jet1_py_obs', b_jet1_py_obs, 'jet1_py_obs/F')
tree.Branch( 'jet1_pz_obs', b_jet1_pz_obs, 'jet1_pz_obs/F')
tree.Branch( 'jet1_pt_obs', b_jet1_pt_obs, 'jet1_pt_obs/F')
tree.Branch( 'jet1_E_obs', b_jet1_E_obs, 'jet1_E_obs/F')
tree.Branch( 'jet1_m_obs', b_jet1_m_obs, 'jet1_m_obs/F')
tree.Branch( 'jet1_btag_obs', b_jet1_btag_obs, 'jet1_btag_obs/F')
tree.Branch( 'jet2_px_obs', b_jet2_px_obs, 'jet2_px_obs/F')
tree.Branch( 'jet2_py_obs', b_jet2_py_obs, 'jet2_py_obs/F')
tree.Branch( 'jet2_pz_obs', b_jet2_pz_obs, 'jet2_pz_obs/F')
tree.Branch( 'jet2_pt_obs', b_jet2_pt_obs, 'jet2_pt_obs/F')
tree.Branch( 'jet2_E_obs', b_jet2_E_obs, 'jet2_E_obs/F')
tree.Branch( 'jet2_m_obs', b_jet2_m_obs, 'jet2_m_obs/F')
tree.Branch( 'jet2_btag_obs', b_jet2_btag_obs, 'jet2_btag_obs/F')
tree.Branch( 'jet3_px_obs', b_jet3_px_obs, 'jet3_px_obs/F')
tree.Branch( 'jet3_py_obs', b_jet3_py_obs, 'jet3_py_obs/F')
tree.Branch( 'jet3_pz_obs', b_jet3_pz_obs, 'jet3_pz_obs/F')
tree.Branch( 'jet3_pt_obs', b_jet3_pt_obs, 'jet3_pt_obs/F')
tree.Branch( 'jet3_E_obs', b_jet3_E_obs, 'jet3_E_obs/F')
tree.Branch( 'jet3_m_obs', b_jet3_m_obs, 'jet3_m_obs/F')
tree.Branch( 'jet3_btag_obs', b_jet3_btag_obs, 'jet3_btag_obs/F')
tree.Branch( 'jet4_px_obs', b_jet4_px_obs, 'jet4_px_obs/F')
tree.Branch( 'jet4_py_obs', b_jet4_py_obs, 'jet4_py_obs/F')
tree.Branch( 'jet4_pz_obs', b_jet4_pz_obs, 'jet4_pz_obs/F')
tree.Branch( 'jet4_pt_obs', b_jet4_pt_obs, 'jet4_pt_obs/F')
tree.Branch( 'jet4_E_obs', b_jet4_E_obs, 'jet4_E_obs/F')
tree.Branch( 'jet4_m_obs', b_jet4_m_obs, 'jet4_m_obs/F')
tree.Branch( 'jet4_btag_obs', b_jet4_btag_obs, 'jet4_btag_obs/F')
tree.Branch( 'jet5_px_obs', b_jet5_px_obs, 'jet5_px_obs/F')
tree.Branch( 'jet5_py_obs', b_jet5_py_obs, 'jet5_py_obs/F')
tree.Branch( 'jet5_pz_obs', b_jet5_pz_obs, 'jet5_pz_obs/F')
tree.Branch( 'jet5_pt_obs', b_jet5_pt_obs, 'jet5_pt_obs/F')
tree.Branch( 'jet5_E_obs', b_jet5_E_obs, 'jet5_E_obs/F')
tree.Branch( 'jet5_m_obs', b_jet5_m_obs, 'jet5_m_obs/F')
tree.Branch( 'jet5_btag_obs', b_jet5_btag_obs, 'jet5_btag_obs/F')
tree.Branch( 'jetmu_px_obs', b_jetmu_px_obs, 'jetmu_px_obs/F')
tree.Branch( 'jetmu_py_obs', b_jetmu_py_obs, 'jetmu_py_obs/F')
tree.Branch( 'jetmu_pz_obs', b_jetmu_pz_obs, 'jetmu_py_obs/F')
tree.Branch( 'jetmu_T0_obs', b_jetmu_T0_obs, 'jetmu_T0_obs/F')
tree.Branch( 'jelep5_ET_obs', b_jetlep_ET_obs, 'jetlep_ET_obs/F')
tree.Branch( 'jetlep_phi_obs', b_jetlep_phi_obs, 'jetlep_phi_obs/F')

tree.Branch( 'W_had_px_true',   b_W_had_px_true,   'W_had_px_true/F' )
tree.Branch( 'W_had_py_true',   b_W_had_py_true,   'W_had_py_true/F' )
tree.Branch( 'W_had_pz_true',   b_W_had_pz_true,   'W_had_pz_true/F' )
tree.Branch( 'W_had_E_true',    b_W_had_E_true,    'W_had_E_true/F' )
tree.Branch( 'W_had_m_true',    b_W_had_m_true,    'W_had_m_true/F' )
tree.Branch( 'W_had_pt_true',   b_W_had_pt_true,   'W_had_pt_true/F' )
tree.Branch( 'W_had_y_true',    b_W_had_y_true,    'W_had_y_true/F' )
tree.Branch( 'W_had_phi_true',  b_W_had_phi_true,  'W_had_phi_true/F' )
tree.Branch( 'b_had_px_true',   b_b_had_px_true,   'b_had_px_true/F' )
tree.Branch( 'b_had_py_true',   b_b_had_py_true,   'b_had_py_true/F' )
tree.Branch( 'b_had_pz_true',   b_b_had_pz_true,   'b_had_pz_true/F' )
tree.Branch( 'b_had_E_true',    b_b_had_E_true,    'b_had_E_true/F' )
tree.Branch( 'b_had_m_true',    b_b_had_m_true,    'b_had_m_true/F' )
tree.Branch( 'b_had_pt_true',   b_b_had_pt_true,   'b_had_pt_true/F' )
tree.Branch( 'b_had_y_true',    b_b_had_y_true,    'b_had_y_true/F' )
tree.Branch( 'b_had_phi_true',  b_b_had_phi_true,  'b_had_phi_true/F' )
tree.Branch( 't_had_px_true',   b_t_had_px_true,   't_had_px_true/F' )
tree.Branch( 't_had_py_true',   b_t_had_py_true,   't_had_py_true/F' )
tree.Branch( 't_had_pz_true',   b_t_had_pz_true,   't_had_pz_true/F' )
tree.Branch( 't_had_E_true',    b_t_had_E_true,    't_had_E_true/F' )
tree.Branch( 't_had_m_true',    b_t_had_m_true,    't_had_m_true/F' )
tree.Branch( 't_had_pt_true',   b_t_had_pt_true,   't_had_pt_true/F' )
tree.Branch( 't_had_y_true',    b_t_had_y_true,    't_had_y_true/F' )
tree.Branch( 't_had_phi_true',  b_t_had_phi_true,  't_had_phi_true/F' )
tree.Branch( 'W_lep_px_true',   b_W_lep_px_true,   'W_lep_px_true/F' )
tree.Branch( 'W_lep_py_true',   b_W_lep_py_true,   'W_lep_py_true/F' )
tree.Branch( 'W_lep_pz_true',   b_W_lep_pz_true,   'W_lep_pz_true/F' )
tree.Branch( 'W_lep_E_true',    b_W_lep_E_true,    'W_lep_E_true/F' )
tree.Branch( 'W_lep_m_true',    b_W_lep_m_true,    'W_lep_m_true/F' )
tree.Branch( 'W_lep_pt_true',   b_W_lep_pt_true,   'W_lep_pt_true/F' )
tree.Branch( 'W_lep_y_true',    b_W_lep_y_true,    'W_lep_y_true/F' )
tree.Branch( 'W_lep_phi_true',  b_W_lep_phi_true,  'W_lep_phi_true/F' )
tree.Branch( 'b_lep_px_true',   b_b_lep_px_true,   'b_lep_px_true/F' )
tree.Branch( 'b_lep_py_true',   b_b_lep_py_true,   'b_lep_py_true/F' )
tree.Branch( 'b_lep_pz_true',   b_b_lep_pz_true,   'b_lep_pz_true/F' )
tree.Branch( 'b_lep_E_true',    b_b_lep_E_true,    'b_lep_E_true/F' )
tree.Branch( 'b_lep_m_true',    b_b_lep_m_true,    'b_lep_m_true/F' )
tree.Branch( 'b_lep_pt_true',   b_b_lep_pt_true,   'b_lep_pt_true/F' )
tree.Branch( 'b_lep_y_true',    b_b_lep_y_true,    'b_lep_y_true/F' )
tree.Branch( 'b_lep_phi_true',  b_b_lep_phi_true,  'b_lep_phi_true/F' )
tree.Branch( 't_lep_px_true',   b_t_lep_px_true,   't_lep_px_true/F' )
tree.Branch( 't_lep_py_true',   b_t_lep_py_true,   't_lep_py_true/F' )
tree.Branch( 't_lep_pz_true',   b_t_lep_pz_true,   't_lep_pz_true/F' )
tree.Branch( 't_lep_E_true',    b_t_lep_E_true,    't_lep_E_true/F' )
tree.Branch( 't_lep_m_true',    b_t_lep_m_true,    't_lep_m_true/F' )
tree.Branch( 't_lep_pt_true',   b_t_lep_pt_true,   't_lep_pt_true/F' )
tree.Branch( 't_lep_y_true',    b_t_lep_y_true,    't_lep_y_true/F' )
tree.Branch( 't_lep_phi_true',  b_t_lep_phi_true,  't_lep_phi_true/F' )

tree.Branch( 'W_had_px_fitted',   b_W_had_px_fitted,   'W_had_px_fitted/F' )
tree.Branch( 'W_had_py_fitted',   b_W_had_py_fitted,   'W_had_py_fitted/F' )
tree.Branch( 'W_had_pz_fitted',   b_W_had_pz_fitted,   'W_had_pz_fitted/F' )
tree.Branch( 'W_had_E_fitted',    b_W_had_E_fitted,    'W_had_E_fitted/F' )
tree.Branch( 'W_had_m_fitted',    b_W_had_m_fitted,    'W_had_m_fitted/F' )
tree.Branch( 'W_had_pt_fitted',   b_W_had_pt_fitted,   'W_had_pt_fitted/F' )
tree.Branch( 'W_had_y_fitted',    b_W_had_y_fitted,    'W_had_y_fitted/F' )
tree.Branch( 'W_had_phi_fitted',  b_W_had_phi_fitted,  'W_had_phi_fitted/F' )
tree.Branch( 'b_had_px_fitted',   b_b_had_px_fitted,   'b_had_px_fitted/F' )
tree.Branch( 'b_had_py_fitted',   b_b_had_py_fitted,   'b_had_py_fitted/F' )
tree.Branch( 'b_had_pz_fitted',   b_b_had_pz_fitted,   'b_had_pz_fitted/F' )
tree.Branch( 'b_had_E_fitted',    b_b_had_E_fitted,    'b_had_E_fitted/F' )
tree.Branch( 'b_had_m_fitted',    b_b_had_m_fitted,    'b_had_m_fitted/F' )
tree.Branch( 'b_had_pt_fitted',   b_b_had_pt_fitted,   'b_had_pt_fitted/F' )
tree.Branch( 'b_had_y_fitted',    b_b_had_y_fitted,    'b_had_y_fitted/F' )
tree.Branch( 'b_had_phi_fitted',  b_b_had_phi_fitted,  'b_had_phi_fitted/F' )
tree.Branch( 't_had_px_fitted',   b_t_had_px_fitted,   't_had_px_fitted/F' )
tree.Branch( 't_had_py_fitted',   b_t_had_py_fitted,   't_had_py_fitted/F' )
tree.Branch( 't_had_pz_fitted',   b_t_had_pz_fitted,   't_had_pz_fitted/F' )
tree.Branch( 't_had_E_fitted',    b_t_had_E_fitted,    't_had_E_fitted/F' )
tree.Branch( 't_had_m_fitted',    b_t_had_m_fitted,    't_had_m_fitted/F' )
tree.Branch( 't_had_pt_fitted',   b_t_had_pt_fitted,   't_had_pt_fitted/F' )
tree.Branch( 't_had_y_fitted',    b_t_had_y_fitted,    't_had_y_fitted/F' )
tree.Branch( 't_had_phi_fitted',  b_t_had_phi_fitted,  't_had_phi_fitted/F' )
tree.Branch( 'W_lep_px_fitted',   b_W_lep_px_fitted,   'W_lep_px_fitted/F' )
tree.Branch( 'W_lep_py_fitted',   b_W_lep_py_fitted,   'W_lep_py_fitted/F' )
tree.Branch( 'W_lep_pz_fitted',   b_W_lep_pz_fitted,   'W_lep_pz_fitted/F' )
tree.Branch( 'W_lep_E_fitted',    b_W_lep_E_fitted,    'W_lep_E_fitted/F' )
tree.Branch( 'W_lep_m_fitted',    b_W_lep_m_fitted,    'W_lep_m_fitted/F' )
tree.Branch( 'W_lep_pt_fitted',   b_W_lep_pt_fitted,   'W_lep_pt_fitted/F' )
tree.Branch( 'W_lep_y_fitted',    b_W_lep_y_fitted,    'W_lep_y_fitted/F' )
tree.Branch( 'W_lep_phi_fitted',  b_W_lep_phi_fitted,  'W_lep_phi_fitted/F' )
tree.Branch( 'b_lep_px_fitted',   b_b_lep_px_fitted,   'b_lep_px_fitted/F' )
tree.Branch( 'b_lep_py_fitted',   b_b_lep_py_fitted,   'b_lep_py_fitted/F' )
tree.Branch( 'b_lep_pz_fitted',   b_b_lep_pz_fitted,   'b_lep_pz_fitted/F' )
tree.Branch( 'b_lep_E_fitted',    b_b_lep_E_fitted,    'b_lep_E_fitted/F' )
tree.Branch( 'b_lep_m_fitted',    b_b_lep_m_fitted,    'b_lep_m_fitted/F' )
tree.Branch( 'b_lep_pt_fitted',   b_b_lep_pt_fitted,   'b_lep_pt_fitted/F' )
tree.Branch( 'b_lep_y_fitted',    b_b_lep_y_fitted,    'b_lep_y_fitted/F' )
tree.Branch( 'b_lep_phi_fitted',  b_b_lep_phi_fitted,  'b_lep_phi_fitted/F' )
tree.Branch( 't_lep_px_fitted',   b_t_lep_px_fitted,   't_lep_px_fitted/F' )
tree.Branch( 't_lep_py_fitted',   b_t_lep_py_fitted,   't_lep_py_fitted/F' )
tree.Branch( 't_lep_pz_fitted',   b_t_lep_pz_fitted,   't_lep_pz_fitted/F' )
tree.Branch( 't_lep_E_fitted',    b_t_lep_E_fitted,    't_lep_E_fitted/F' )
tree.Branch( 't_lep_m_fitted',    b_t_lep_m_fitted,    't_lep_m_fitted/F' )
tree.Branch( 't_lep_pt_fitted',   b_t_lep_pt_fitted,   't_lep_pt_fitted/F' )
tree.Branch( 't_lep_y_fitted',    b_t_lep_y_fitted,    't_lep_y_fitted/F' )
tree.Branch( 't_lep_phi_fitted',  b_t_lep_phi_fitted,  't_lep_phi_fitted/F' )

################################################################################
# SET COUNTERS
# list of number of events best matched to 1,2,3 jets respectively.
W_had_jets, W_had_total_cuts = [0., 0., 0.] , [0., 0., 0.] 

# Counters to make tally number of events that pass cuts
W_had_m_cuts, W_had_pT_cuts, W_had_dist_cuts = [0., 0., 0.] , [0., 0., 0.]  , [0., 0., 0.] 
W_lep_total_cuts, W_lep_ET_cuts, W_lep_dist_cuts = 0., 0., 0.
b_had_pT_cuts, b_had_dist_cuts, b_had_total_cuts = 0., 0., 0.
b_lep_pT_cuts, b_lep_dist_cuts, b_lep_total_cuts = 0., 0., 0.

good_events = 0.

################################################################################
# POPULATE TREE
print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
    w = 1
    jets_n  = event_info[i][3]
    bjets_n = event_info[i][4]

        
    W_had_true   = MakeP4( y_true_W_had[i], m_W, representation)
    W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W, representation)

    W_lep_true   = MakeP4( y_true_W_lep[i], m_W , representation)
    W_lep_fitted = MakeP4( y_fitted_W_lep[i],  m_W, representation)

    b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
    b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b , representation)

    b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
    b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b, representation)

    t_had_true   = MakeP4( y_true_t_had[i], m_t , representation)
    t_had_fitted = MakeP4( y_fitted_t_had[i],  m_t , representation)

    t_lep_true   = MakeP4( y_true_t_lep[i], m_t , representation)
    t_lep_fitted = MakeP4( y_fitted_t_lep[i],  m_t, representation)

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
    b_had_dist_true = b_lep_dist_true = W_had_dist_true = 1000000

    # Perform jet matching for the bs, all jets, b-tagged and not b-tagged should be considered.
    for k in range(len(jets)): # loop through each jet to find the minimum distance for each particle
        # For bs:
        b_had_d_true = find_dist(b_had_true, jets[k])
        if b_had_d_true < b_had_dist_true:
            b_had_dist_true = b_had_d_true
            closest_b_had = jets[k]
        b_lep_d_true = find_dist(b_lep_true, jets[k])
        if b_lep_d_true < b_lep_dist_true:
            b_lep_dist_true = b_lep_d_true
            closest_b_lep = jets[k]

    good_jets = jets[:]
    if (b_tagging > 0):
        for m in range(len(jets)):
            # if don't include any b tagged jets and jet is b tagged OR
            # if only considering b tagged jets and jet is not b tagged
            if (b_tagging == 1 and jet_list[m, i, 5]) or (b_tagging == 2 and not jet_list[m,i,5]):
                good_jets.remove(jets[m])
    # If there are no jets remaining in good_jets, then skip this event. Don't populate histograms.
    if not good_jets:
        continue
    
    # Consider best two jets first.
    if (len(good_jets) >= 2):
        for k in range(len(good_jets)):
            # if good_jets only contains one element, loop is skipped since range would be (1,1)
            for j in range(k + 1, len(good_jets)):                  
                sum_vect = good_jets[k] + good_jets[j] 
                W_had_d_true = find_dist(W_had_true, sum_vect)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    closest_W_had = sum_vect
        W_had_true_obs_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
        num_jets = 1
    
    # If the best double jet doesn't pass cuts, then consider three jets.
    if (len(good_jets) >= 3) and (closest_W_had.M() <= W_had_m_cutoff[0] \
        or closest_W_had.M() >= W_had_m_cutoff[1] or W_had_true_obs_pT_diff <= W_had_pT_cutoff[0] \
        or W_had_true_obs_pT_diff >= W_had_pT_cutoff[1] \
        or W_had_dist_true >= W_had_dist_cutoff[1]):
        # Reset maximum eta-phi distance.
        W_had_dist_true = 1000000
        for k in range(len(good_jets)):
            for j in range(k + 1, len(good_jets)):     
                for l in range(j+1, len(good_jets)):
                    sum_vect = good_jets[k] + good_jets[j] + good_jets[l]
                    # Calculate eta-phi distance for current jet combo.
                    W_had_d_true = find_dist(W_had_true, sum_vect)
                    # Compare current distance to current minimum distance and update if lower.
                    if W_had_d_true < W_had_dist_true:
                        W_had_dist_true = W_had_d_true
                        closest_W_had = sum_vect
        # Calculate true - observed pT difference for the best triple jet
        W_had_true_obs_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
        num_jets = 2

    # if there is only one jet in the list or previous matches don't pass cutoff conditions, find a single jet match
    if (len(good_jets) == 1) or ((closest_W_had.M() <= W_had_m_cutoff[0] or closest_W_had.M() >= W_had_m_cutoff[1]) \
        or (W_had_true_obs_pT_diff <= W_had_pT_cutoff[0] or W_had_true_obs_pT_diff >= W_had_pT_cutoff[1])\
        or W_had_dist_true >= W_had_dist_cutoff[1]):
        W_had_dist_true = 1000000
        # Single jets
        for k in range(len(good_jets)):
            sum_vect = good_jets[k]    
            W_had_d_true = find_dist(W_had_true, sum_vect)
            if W_had_d_true < W_had_dist_true:
                W_had_dist_true = W_had_d_true
                closest_W_had = sum_vect
        # Only calculate difference for best single jet.
        W_had_true_obs_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
        num_jets = 0

    # Special calculations for the observed leptonic W assuming massless daughters.  
    muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]]
    # Observed neutrino transverse momentum from missing energy as [x, y].
    nu_pT_obs = [jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] 
    W_lep_Px_observed = muon_pT_obs[0] + nu_pT_obs[0]
    W_lep_Py_observed = muon_pT_obs[1] + nu_pT_obs[1]
    
    # Calculate the distance between true and observed phi.
    lep_phi = np.arctan2(W_lep_Py_observed, W_lep_Px_observed)
    W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
    # Calculate transverse energy assuming daughter particles are massless
    W_lep_ET_observed = np.sqrt( W_lep_Px_observed**2 + W_lep_Py_observed**2)
    W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed
    # Calculate the transverse mass
    obs_daughter_angle = np.arccos( np.dot(muon_pT_obs, nu_pT_obs) / norm(muon_pT_obs) / norm(nu_pT_obs) )
    met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(obs_daughter_angle))) 

    # Compare hadronic t distances
    t_had_jets = closest_b_had + closest_W_had
    t_had_dist_true = find_dist(t_had_true, t_had_jets)
    t_had_true_obs_pT_diff = t_had_true.Pt() - t_had_jets.Pt()
    
    # Compare leptonic t distances
    t_lep_x = W_lep_Px_observed + closest_b_lep.Px()
    t_lep_y = W_lep_Py_observed + closest_b_lep.Py()
    obs_t_phi = np.arctan2(t_lep_y, t_lep_x)
    t_lep_dist_true = np.abs( min( np.abs(t_lep_true.Phi()-obs_t_phi), 2*np.pi-np.abs(t_lep_true.Phi() - obs_t_phi) ) )
    t_lep_pT_observed = np.sqrt( t_lep_x**2 + t_lep_y**2)
    t_lep_true_obs_pT_diff = t_lep_true.Et() - t_lep_pT_observed

    # b quark calculations
    b_had_true_obs_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
    b_lep_true_obs_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()


    ############################################## check whether each event passes cuts #################################################
    # counter for hadronic W
    # Update tally for which jet combination is the closest
    W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
    W_had_pT_cut = (W_had_true_obs_pT_diff >= W_had_pT_cutoff[0] and W_had_true_obs_pT_diff <= W_had_pT_cutoff[1])
    W_had_dist_cut = (W_had_dist_true <= W_had_dist_cutoff[1]) 
    # All W_had cuts must be satisfied simultaneously.
    good_W_had = (W_had_m_cut and W_had_pT_cut and W_had_dist_cut)

    W_had_jets[num_jets] += 1.
    W_had_total_cuts[num_jets] += good_W_had
    W_had_m_cuts[num_jets] += W_had_m_cut
    W_had_pT_cuts[num_jets] += W_had_pT_cut
    W_had_dist_cuts[num_jets] += W_had_dist_cut

    # counter for lep W
    W_lep_ET_cut = (W_lep_ET_diff >= W_lep_ET_cutoff[0] and W_lep_ET_diff <= W_lep_ET_cutoff[1])
    W_lep_dist_cut = (W_lep_dist_true <= W_lep_dist_cutoff[1]) 
    good_W_lep = (W_lep_ET_cut and W_lep_dist_cut)

    W_lep_total_cuts += good_W_lep
    W_lep_ET_cuts += W_lep_ET_cut
    W_lep_dist_cuts += W_lep_dist_cut

    # counter for hadronic b
    b_had_pT_cut = (b_had_true_obs_pT_diff >= b_had_pT_cutoff[0] and b_had_true_obs_pT_diff <= b_had_pT_cutoff[1])
    b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
    good_b_had = (b_had_pT_cut and b_had_dist_cut)

    b_had_total_cuts += good_b_had
    b_had_pT_cuts += b_had_pT_cut
    b_had_dist_cuts += b_had_dist_cut

    # counter for leptonic b
    b_lep_pT_cut = (b_lep_true_obs_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_obs_pT_diff <= b_lep_pT_cutoff[1])
    b_lep_dist_cut = (b_lep_dist_true <= b_lep_dist_cutoff[1]) 
    good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

    b_lep_total_cuts += good_b_lep
    b_lep_pT_cuts += b_lep_pT_cut
    b_lep_dist_cuts += b_lep_dist_cut

    # Good events must pass cuts on all partons.
    good_event = (good_b_had and good_b_lep and good_W_had and good_W_lep)
    good_events += good_event

    ################################################# fill branches #################################################

    b_eventNumber[0] = int(event_info[i][0])
    b_runNumber[0]   = int(event_info[i][1])
    b_weight_mc[0]   = float(event_info[i][2])

    if ((event_type == "bad") and (not good_event)) or (event_type == "good" and good_event) or event_type == "all":
        # jets
        b_W_had_px_obs[0] = closest_W_had.Px()
        b_W_had_py_obs[0]   = closest_W_had.Py()
        b_W_had_pz_obs[0]   = closest_W_had.Pz()
        b_W_had_E_obs[0]    = closest_W_had.E()
        b_W_had_m_obs[0] = closest_W_had.M()
        b_W_had_pt_obs[0]   = closest_W_had.Pt()
        b_W_had_y_obs[0]    = closest_W_had.Rapidity()
        b_W_had_phi_obs[0]  = closest_W_had.Phi()
        b_W_had_num_jets[0]  = num_jets + 1
        b_b_had_px_obs[0]   = closest_b_had.Px()
        b_b_had_py_obs[0]   = closest_b_had.Py()
        b_b_had_pz_obs[0]   = closest_b_had.Pz()
        b_b_had_E_obs [0]   = closest_b_had.E()
        b_b_had_m_obs [0]   = closest_b_had.M()
        b_b_had_pt_obs[0]   = closest_b_had.Pt()
        b_b_had_y_obs [0]   = closest_b_had.Rapidity()
        b_b_had_phi_obs[0]  = closest_b_had.Phi()

        b_t_had_px_obs[0]   = t_had_jets.Px()
        b_t_had_py_obs[0]   = t_had_jets.Py()
        b_t_had_pz_obs[0]   = t_had_jets.Pz()
        b_t_had_E_obs [0]   = t_had_jets.E()
        b_t_had_m_obs [0]   = t_had_jets.M()
        b_t_had_pt_obs[0]   = t_had_jets.Pt()
        b_t_had_y_obs [0]   = t_had_jets.Rapidity()
        b_t_had_phi_obs[0]  = t_had_jets.Phi()

        b_W_lep_px_obs[0]   = W_lep_Px_observed
        b_W_lep_py_obs[0]   = W_lep_Py_observed
        b_W_lep_Et_obs [0]  = W_lep_ET_observed
        b_W_lep_phi_obs[0]  = lep_phi
        b_b_lep_px_obs[0]   = closest_b_lep.Px()
        b_b_lep_py_obs[0]   = closest_b_lep.Py()
        b_b_lep_pz_obs[0]   = closest_b_lep.Pz()
        b_b_lep_E_obs [0]   = closest_b_lep.E()
        b_b_lep_m_obs [0]   = closest_b_lep.M()
        b_b_lep_pt_obs[0]   = closest_b_lep.Pt()
        b_b_lep_y_obs [0]   = closest_b_lep.Rapidity()
        b_b_lep_phi_obs[0]  = closest_b_lep.Phi()

        b_t_lep_px_obs[0]   = t_lep_x
        b_t_lep_py_obs[0]   = t_lep_y
        b_t_lep_pt_obs [0]  = t_lep_pT_observed
        b_t_lep_phi_obs[0]  = obs_t_phi


        b_jet1_px_obs[0]   = jet_1_vect.Px()
        b_jet1_py_obs[0]   = jet_1_vect.Py()
        b_jet1_pz_obs[0]   = jet_1_vect.Pz()
        b_jet1_pt_obs[0]   = jet_1_vect.Pt()
        b_jet1_E_obs [0]   = jet_1_vect.E()
        b_jet1_m_obs [0]   = jet_1_vect.M()
        b_jet1_btag_obs[0]   = jet_1[i][5]
        b_jet2_px_obs[0]   = jet_2_vect.Px()
        b_jet2_py_obs[0]   = jet_2_vect.Py()
        b_jet2_pz_obs[0]   = jet_2_vect.Pz()
        b_jet2_pt_obs[0]   = jet_2_vect.Pt()
        b_jet2_E_obs [0]   = jet_2_vect.E()
        b_jet2_m_obs [0]   = jet_2_vect.M()
        b_jet2_btag_obs[0]   = jet_2[i][5]
        b_jet3_px_obs[0]   = jet_3_vect.Px()
        b_jet3_py_obs[0]   = jet_3_vect.Py()
        b_jet3_pz_obs[0]   = jet_3_vect.Pz()
        b_jet3_pt_obs[0]   = jet_3_vect.Pt()
        b_jet3_E_obs [0]   = jet_3_vect.E()
        b_jet3_m_obs [0]   = jet_3_vect.M()
        b_jet3_btag_obs[0]   = jet_3[i][5]
        b_jet4_px_obs[0]   = jet_4_vect.Px()
        b_jet4_py_obs[0]   = jet_4_vect.Py()
        b_jet4_pz_obs[0]   = jet_4_vect.Pz()
        b_jet4_pt_obs[0]   = jet_4_vect.Pt()
        b_jet4_E_obs [0]   = jet_4_vect.E()
        b_jet4_m_obs [0]   = jet_4_vect.M()
        b_jet4_btag_obs[0]   = jet_4[i][5]
        b_jet5_px_obs[0]   = jet_5_vect.Px()
        b_jet5_py_obs[0]   = jet_5_vect.Py()
        b_jet5_pz_obs[0]   = jet_5_vect.Pz()
        b_jet5_pt_obs[0]   = jet_5_vect.Pt()
        b_jet5_E_obs [0]   = jet_5_vect.E()
        b_jet5_m_obs [0]   = jet_5_vect.M()
        b_jet5_btag_obs[0]   = jet_5[i][5]
        b_jetmu_px_obs[0]   = jet_mu[i][0]
        b_jetmu_py_obs[0]   = jet_mu[i][1]
        b_jetmu_pz_obs[0]   = jet_mu[i][2]
        b_jetmu_T0_obs [0]   = jet_mu[i][3]
        b_jetlep_ET_obs [0]   = jet_mu[i][4]
        b_jetlep_phi_obs[0]   = jet_mu[i][5]


        # true

        b_W_had_px_true[0]  = W_had_true.Px()
        b_W_had_py_true[0]  = W_had_true.Py()
        b_W_had_pz_true[0]  = W_had_true.Pz()
        b_W_had_E_true[0]   = W_had_true.E()
        b_W_had_m_true[0]   = W_had_true.M()
        b_W_had_pt_true[0]  = W_had_true.Pt()
        b_W_had_y_true[0]   = W_had_true.Rapidity()
        b_W_had_phi_true[0] = W_had_true.Phi()

        b_b_had_px_true[0]  = b_had_true.Px()
        b_b_had_py_true[0]  = b_had_true.Py()
        b_b_had_pz_true[0]  = b_had_true.Pz()
        b_b_had_E_true[0]   = b_had_true.E()
        b_b_had_m_true[0]   = b_had_true.M()
        b_b_had_pt_true[0]  = b_had_true.Pt()
        b_b_had_y_true[0]   = b_had_true.Rapidity()
        b_b_had_phi_true[0] = b_had_true.Phi()

        b_t_had_px_true[0]  = t_had_true.Px()
        b_t_had_py_true[0]  = t_had_true.Py()
        b_t_had_pz_true[0]  = t_had_true.Pz()
        b_t_had_E_true[0]   = t_had_true.E()
        b_t_had_m_true[0]   = t_had_true.M()
        b_t_had_pt_true[0]  = t_had_true.Pt()
        b_t_had_y_true[0]   = t_had_true.Rapidity()
        b_t_had_phi_true[0] = t_had_true.Phi()

        b_W_lep_px_true[0]  = W_lep_true.Px()
        b_W_lep_py_true[0]  = W_lep_true.Py()
        b_W_lep_pz_true[0]  = W_lep_true.Pz()
        b_W_lep_E_true[0]   = W_lep_true.E()
        b_W_lep_m_true[0]   = W_lep_true.M()
        b_W_lep_pt_true[0]  = W_lep_true.Pt()
        b_W_lep_y_true[0]   = W_lep_true.Rapidity()
        b_W_lep_phi_true[0] = W_lep_true.Phi()

        b_b_lep_px_true[0]  = b_lep_true.Px()
        b_b_lep_py_true[0]  = b_lep_true.Py()
        b_b_lep_pz_true[0]  = b_lep_true.Pz()
        b_b_lep_E_true[0]   = b_lep_true.E()
        b_b_lep_m_true[0]   = b_lep_true.M()
        b_b_lep_pt_true[0]  = b_lep_true.Pt()
        b_b_lep_y_true[0]   = b_lep_true.Rapidity()
        b_b_lep_phi_true[0] = b_lep_true.Phi()

        b_t_lep_px_true[0]  = t_lep_true.Px()
        b_t_lep_py_true[0]  = t_lep_true.Py()
        b_t_lep_pz_true[0]  = t_lep_true.Pz()
        b_t_lep_E_true[0]   = t_lep_true.E()
        b_t_lep_m_true[0]   = t_lep_true.M()
        b_t_lep_pt_true[0]  = t_lep_true.Pt()
        b_t_lep_y_true[0]   = t_lep_true.Rapidity()
        b_t_lep_phi_true[0] = t_lep_true.Phi()

        # fitted

        b_W_had_px_fitted[0]  = W_had_fitted.Px()
        b_W_had_py_fitted[0]  = W_had_fitted.Py()
        b_W_had_pz_fitted[0]  = W_had_fitted.Pz()
        b_W_had_E_fitted[0]   = W_had_fitted.E()
        b_W_had_m_fitted[0]   = W_had_fitted.M()
        b_W_had_pt_fitted[0]  = W_had_fitted.Pt()
        b_W_had_y_fitted[0]   = W_had_fitted.Rapidity()
        b_W_had_phi_fitted[0] = W_had_fitted.Phi()

        b_b_had_px_fitted[0]  = b_had_fitted.Px()
        b_b_had_py_fitted[0]  = b_had_fitted.Py()
        b_b_had_pz_fitted[0]  = b_had_fitted.Pz()
        b_b_had_E_fitted[0]   = b_had_fitted.E()
        b_b_had_m_fitted[0]   = b_had_fitted.M()
        b_b_had_pt_fitted[0]  = b_had_fitted.Pt()
        b_b_had_y_fitted[0]   = b_had_fitted.Rapidity()
        b_b_had_phi_fitted[0] = b_had_fitted.Phi()

        b_t_had_px_fitted[0]  = t_had_fitted.Px()
        b_t_had_py_fitted[0]  = t_had_fitted.Py()
        b_t_had_pz_fitted[0]  = t_had_fitted.Pz()
        b_t_had_E_fitted[0]   = t_had_fitted.E()
        b_t_had_m_fitted[0]   = t_had_fitted.M()
        b_t_had_pt_fitted[0]  = t_had_fitted.Pt()
        b_t_had_y_fitted[0]   = t_had_fitted.Rapidity()
        b_t_had_phi_fitted[0] = t_had_fitted.Phi()

        b_W_lep_px_fitted[0]  = W_lep_fitted.Px()
        b_W_lep_py_fitted[0]  = W_lep_fitted.Py()
        b_W_lep_pz_fitted[0]  = W_lep_fitted.Pz()
        b_W_lep_E_fitted[0]   = W_lep_fitted.E()
        b_W_lep_m_fitted[0]   = W_lep_fitted.M()
        b_W_lep_pt_fitted[0]  = W_lep_fitted.Pt()
        b_W_lep_y_fitted[0]   = W_lep_fitted.Rapidity()
        b_W_lep_phi_fitted[0] = W_lep_fitted.Phi()

        b_b_lep_px_fitted[0]  = b_lep_fitted.Px()
        b_b_lep_py_fitted[0]  = b_lep_fitted.Py()
        b_b_lep_pz_fitted[0]  = b_lep_fitted.Pz()
        b_b_lep_E_fitted[0]   = b_lep_fitted.E()
        b_b_lep_m_fitted[0]   = b_lep_fitted.M()
        b_b_lep_pt_fitted[0]  = b_lep_fitted.Pt()
        b_b_lep_y_fitted[0]   = b_lep_fitted.Rapidity()
        b_b_lep_phi_fitted[0] = b_lep_fitted.Phi()

        b_t_lep_px_fitted[0]  = t_lep_fitted.Px()
        b_t_lep_py_fitted[0]  = t_lep_fitted.Py()
        b_t_lep_pz_fitted[0]  = t_lep_fitted.Pz()
        b_t_lep_E_fitted[0]   = t_lep_fitted.E()
        b_t_lep_m_fitted[0]   = t_lep_fitted.M()
        b_t_lep_pt_fitted[0]  = t_lep_fitted.Pt()
        b_t_lep_y_fitted[0]   = t_lep_fitted.Rapidity()
        b_t_lep_phi_fitted[0] = t_lep_fitted.Phi()

        tree.Fill()

################################################################################
# CLOSE PROGRAM
ofile.Write()
ofile.Close()

print("Finished. Saved output file:", ofilename)

if event_type == "good":
    # Print data regarding percentage of each class of event
    print('Total number of events: {} \n'.format(n_events))
    print('NOTE: some percentages do not reach 100%, as events where no Hadronic W can be matched after removing the b-tagged jets are skipped (all jets are b-tagged)')
    print('\n==================================================================\n')
    print('Cut Criteria')
    print('Hadronic W, mass: {}, pT: {}, distance: {}'.format(W_had_m_cutoff, W_had_pT_cutoff, W_had_dist_cutoff))
    print('Leptonic W, E_T: {}, dist: {}'.format(W_lep_ET_cutoff, W_lep_dist_cutoff))
    print('Hadronic b, pT: {}, distance: {}'.format(b_had_pT_cutoff, b_had_dist_cutoff))
    print('Leptonic b, pT: {}, distance: {}'.format(b_lep_pT_cutoff, b_lep_dist_cutoff))
    print('\n==================================================================\n')

    print("Breakdown of total Hadronic Ws matched to 1, 2, and 3 jets, before applying cuts on events matched to 1 jet:")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_jets[0]/n_events, int(W_had_jets[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_jets[1]/n_events, int(W_had_jets[1])))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_jets[2]/n_events, int(W_had_jets[2])))

    print("Number of events satisfying all hadronic W cut criteria, as a percentage of their respective categories before applying cuts:")
    print('{}% Total Hadronic Ws within cuts, {} events'.format(100.*sum(W_had_total_cuts)/n_events, int(sum(W_had_total_cuts))))
    print('{}% 1 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[0]/W_had_jets[0], int(W_had_total_cuts[0]), int(W_had_jets[0])))
    print('{}% 2 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[1]/W_had_jets[1], int(W_had_total_cuts[1]), int(W_had_jets[1])))
    print('{}% 3 jet Hadronic W, {} events, out of {}\n'.format(100.*W_had_total_cuts[2]/W_had_jets[2], int(W_had_total_cuts[2]), int(W_had_jets[2])))

    print("Breakdown of total Hadronic Ws matched to 1, 2, and 3 jets after cuts are applied: ")
    print('{}% 1 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[0]/sum(W_had_total_cuts), int(W_had_total_cuts[0]), int(sum(W_had_total_cuts))))
    print('{}% 2 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[1]/sum(W_had_total_cuts), int(W_had_total_cuts[1]), int(sum(W_had_total_cuts))))
    print('{}% 3 jet Hadronic W, {} events, out of {}\n'.format(100.*W_had_total_cuts[2]/sum(W_had_total_cuts), int(W_had_total_cuts[2]), int(sum(W_had_total_cuts))))

    print("Number of events satisfying hadronic W mass cut criteria")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_m_cuts[0]/W_had_jets[0], int(W_had_m_cuts[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_m_cuts[1]/W_had_jets[1], int(W_had_m_cuts[1])))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_m_cuts[2]/W_had_jets[2], int(W_had_m_cuts[2])))
    print("Number of events satisfying hadronic W pT cut criteria")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_pT_cuts[0]/W_had_jets[0], int(W_had_pT_cuts[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_pT_cuts[1]/W_had_jets[1], int(W_had_pT_cuts[1])))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_pT_cuts[2]/W_had_jets[2], int(W_had_pT_cuts[2])))
    print("Number of events satisfying hadronic W distance cut criteria")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[0]/W_had_jets[0], int(W_had_dist_cuts[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[1]/W_had_jets[1], int(W_had_dist_cuts[1])))
    print('{}% 3 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[2]/W_had_jets[2], int(W_had_dist_cuts[2])))
    print('\n==================================================================\n')
    print("Number of events satisfying all leptonic W cut criteria")
    print('{}% , {} events\n'.format(100.*W_lep_total_cuts/n_events, int(W_lep_total_cuts)))
    print("Number of events satisfying leptonic W ET cut criteria")
    print('{}%, {} events'.format(100.*W_lep_ET_cuts/n_events, int(W_lep_ET_cuts)))
    print("Number of events satisfying leptonic W distance cut criteria")
    print('{}%, {} events'.format(100.*W_lep_dist_cuts/n_events, int(W_lep_dist_cuts)))
    print('\n==================================================================\n')
    print("Number of events satisfying all hadronic b cut criteria")
    print('{}% , {} events\n'.format(100.*b_had_total_cuts/n_events, int(b_had_total_cuts)))
    print("Number of events satisfying hadronic b pT cut criteria")
    print('{}%, {} events'.format(100.*b_had_pT_cuts/n_events, int(b_had_pT_cuts)))
    print("Number of events satisfying hadronic b distance cut criteria")
    print('{}%, {} events'.format(100.*b_had_dist_cuts/n_events, int(b_had_dist_cuts)))
    print('\n==================================================================\n')
    print("Number of events satisfying all leptonic b cut criteria")
    print('{}% , {} events\n'.format(100.*b_lep_total_cuts/n_events, int(b_lep_total_cuts)))
    print("Number of events satisfying leptonic b pT cut criteria")
    print('{}%, {} events'.format(100.*b_lep_pT_cuts/n_events, int(b_lep_pT_cuts)))
    print("Number of events satisfying leptonic b distance cut criteria")
    print('{}%, {} events'.format(100.*b_lep_dist_cuts/n_events, int(b_lep_dist_cuts)))
    print('\n==================================================================\n')
    print("Events satisfying cut all cut criteria for all partons")
    print('{}%, {} events\n\n'.format(100.*good_events/n_events, int(good_events)))