#!/usr/bin/env python
from ROOT import *
import numpy as np
from AngryTops.features import *
import array
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, undo_scaling

# write to tree file

################################################################################
# CONSTANTS
training_dir = "../CheckPoints/Summer/May21/"
output_dir = "../CheckPoints/Summer/May21/"
representation = "pxpypzEM"
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

ALL = 0
NONE = 1
ONLY = 2
b_tagging = NONE   

################################################################################
# load data

print("INFO: fitting ttbar decay chain...")
predictions = np.load(training_dir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']

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

ofilename = "{}/predictions_May21.root".format(output_dir)
# Open output file
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

b_jet1_px_obs   = array.array( 'f', [ -1.] )
b_jet1_py_obs   = array.array( 'f', [ -1.] )
b_jet1_pz_obs   = array.array( 'f', [ -1.] )
b_jet1_pt_obs   = array.array( 'f', [ -1.] )
b_jet1_E_obs    = array.array( 'f', [ -1.] )
b_jet1_m_obs    = array.array( 'f', [ -1.] )
b_jet1_btag_obs   = array.array( 'f', [ -1.] )
b_jet2_px_obs   = array.array( 'f', [ -1.] )
b_jet2_py_obs   = array.array( 'f', [ -1.] )
b_jet2_pz_obs   = array.array( 'f', [ -1.] )
b_jet2_pt_obs   = array.array( 'f', [ -1.] )
b_jet2_E_obs    = array.array( 'f', [ -1.] )
b_jet2_m_obs    = array.array( 'f', [ -1.] )
b_jet2_btag_obs   = array.array( 'f', [ -1.] )
b_jet3_px_obs   = array.array( 'f', [ -1.] )
b_jet3_py_obs   = array.array( 'f', [ -1.] )
b_jet3_pz_obs   = array.array( 'f', [ -1.] )
b_jet3_pt_obs   = array.array( 'f', [ -1.] )
b_jet3_E_obs    = array.array( 'f', [ -1.] )
b_jet3_m_obs    = array.array( 'f', [ -1.] )
b_jet3_btag_obs   = array.array( 'f', [ -1.] )
b_jet4_px_obs   = array.array( 'f', [ -1.] )
b_jet4_py_obs   = array.array( 'f', [ -1.] )
b_jet4_pz_obs   = array.array( 'f', [ -1.] )
b_jet4_pt_obs   = array.array( 'f', [ -1.] )
b_jet4_E_obs    = array.array( 'f', [ -1.] )
b_jet4_m_obs    = array.array( 'f', [ -1.] )
b_jet4_btag_obs   = array.array( 'f', [ -1.] )
b_jet5_px_obs   = array.array( 'f', [ -1.] )
b_jet5_py_obs   = array.array( 'f', [ -1.] )
b_jet5_pz_obs   = array.array( 'f', [ -1.] )
b_jet5_pt_obs   = array.array( 'f', [ -1.] )
b_jet5_E_obs    = array.array( 'f', [ -1.] )
b_jet5_m_obs    = array.array( 'f', [ -1.] )
b_jet5_btag_obs   = array.array( 'f', [ -1.] )
b_jetmu_px_obs   = array.array( 'f', [ -1.] )
b_jetmu_py_obs   = array.array( 'f', [ -1.] )
b_jetmu_pz_obs   = array.array( 'f', [ -1.] )
b_jetmu_T0_obs    = array.array( 'f', [ -1.] )
b_jetlep_ET_obs    = array.array( 'f', [ -1.] )
b_jetlep_phi_obs   = array.array( 'f', [ -1.] )

b_W_had_px_true   = array.array( 'f', [ -1.] )
b_W_had_py_true   = array.array( 'f', [ -1.] )
b_W_had_pz_true   = array.array( 'f', [ -1.] )
b_W_had_E_true    = array.array( 'f', [ -1.] )
b_W_had_m_true    = array.array( 'f', [ -1.] )
b_W_had_pt_true   = array.array( 'f', [ -1.] )
b_W_had_y_true    = array.array( 'f', [ -1.] )
b_W_had_phi_true  = array.array( 'f', [ -1.] )
b_b_had_px_true   = array.array( 'f', [ -1.] )
b_b_had_py_true   = array.array( 'f', [ -1.] )
b_b_had_pz_true   = array.array( 'f', [ -1.] )
b_b_had_E_true    = array.array( 'f', [ -1.] )
b_b_had_m_true    = array.array( 'f', [ -1.] )
b_b_had_pt_true   = array.array( 'f', [ -1.] )
b_b_had_y_true    = array.array( 'f', [ -1.] )
b_b_had_phi_true  = array.array( 'f', [ -1.] )
b_t_had_px_true   = array.array( 'f', [ -1.] )
b_t_had_py_true   = array.array( 'f', [ -1.] )
b_t_had_pz_true   = array.array( 'f', [ -1.] )
b_t_had_E_true    = array.array( 'f', [ -1.] )
b_t_had_m_true    = array.array( 'f', [ -1.] )
b_t_had_pt_true   = array.array( 'f', [ -1.] )
b_t_had_y_true    = array.array( 'f', [ -1.] )
b_t_had_phi_true  = array.array( 'f', [ -1.] )
b_W_lep_px_true   = array.array( 'f', [ -1.] )
b_W_lep_py_true   = array.array( 'f', [ -1.] )
b_W_lep_pz_true   = array.array( 'f', [ -1.] )
b_W_lep_E_true    = array.array( 'f', [ -1.] )
b_W_lep_m_true    = array.array( 'f', [ -1.] )
b_W_lep_pt_true   = array.array( 'f', [ -1.] )
b_W_lep_y_true    = array.array( 'f', [ -1.] )
b_W_lep_phi_true  = array.array( 'f', [ -1.] )
b_b_lep_px_true   = array.array( 'f', [ -1.] )
b_b_lep_py_true   = array.array( 'f', [ -1.] )
b_b_lep_pz_true   = array.array( 'f', [ -1.] )
b_b_lep_E_true    = array.array( 'f', [ -1.] )
b_b_lep_m_true    = array.array( 'f', [ -1.] )
b_b_lep_pt_true   = array.array( 'f', [ -1.] )
b_b_lep_y_true    = array.array( 'f', [ -1.] )
b_b_lep_phi_true  = array.array( 'f', [ -1.] )
b_t_lep_px_true   = array.array( 'f', [ -1.] )
b_t_lep_py_true   = array.array( 'f', [ -1.] )
b_t_lep_pz_true   = array.array( 'f', [ -1.] )
b_t_lep_E_true    = array.array( 'f', [ -1.] )
b_t_lep_m_true    = array.array( 'f', [ -1.] )
b_t_lep_pt_true   = array.array( 'f', [ -1.] )
b_t_lep_y_true    = array.array( 'f', [ -1.] )
b_t_lep_phi_true  = array.array( 'f', [ -1.] )

b_W_had_px_fitted   = array.array( 'f', [ -1.] )
b_W_had_py_fitted   = array.array( 'f', [ -1.] )
b_W_had_pz_fitted   = array.array( 'f', [ -1.] )
b_W_had_E_fitted    = array.array( 'f', [ -1.] )
b_W_had_m_fitted    = array.array( 'f', [ -1.] )
b_W_had_pt_fitted   = array.array( 'f', [ -1.] )
b_W_had_y_fitted    = array.array( 'f', [ -1.] )
b_W_had_phi_fitted  = array.array( 'f', [ -1.] )
b_b_had_px_fitted   = array.array( 'f', [ -1.] )
b_b_had_py_fitted   = array.array( 'f', [ -1.] )
b_b_had_pz_fitted   = array.array( 'f', [ -1.] )
b_b_had_E_fitted    = array.array( 'f', [ -1.] )
b_b_had_m_fitted    = array.array( 'f', [ -1.] )
b_b_had_pt_fitted   = array.array( 'f', [ -1.] )
b_b_had_y_fitted    = array.array( 'f', [ -1.] )
b_b_had_phi_fitted  = array.array( 'f', [ -1.] )
b_t_had_px_fitted   = array.array( 'f', [ -1.] )
b_t_had_py_fitted   = array.array( 'f', [ -1.] )
b_t_had_pz_fitted   = array.array( 'f', [ -1.] )
b_t_had_E_fitted    = array.array( 'f', [ -1.] )
b_t_had_m_fitted    = array.array( 'f', [ -1.] )
b_t_had_pt_fitted   = array.array( 'f', [ -1.] )
b_t_had_y_fitted    = array.array( 'f', [ -1.] )
b_t_had_phi_fitted  = array.array( 'f', [ -1.] )
b_W_lep_px_fitted   = array.array( 'f', [ -1.] )
b_W_lep_py_fitted   = array.array( 'f', [ -1.] )
b_W_lep_pz_fitted   = array.array( 'f', [ -1.] )
b_W_lep_E_fitted    = array.array( 'f', [ -1.] )
b_W_lep_m_fitted    = array.array( 'f', [ -1.] )
b_W_lep_pt_fitted   = array.array( 'f', [ -1.] )
b_W_lep_y_fitted    = array.array( 'f', [ -1.] )
b_W_lep_phi_fitted  = array.array( 'f', [ -1.] )
b_b_lep_px_fitted   = array.array( 'f', [ -1.] )
b_b_lep_py_fitted   = array.array( 'f', [ -1.] )
b_b_lep_pz_fitted   = array.array( 'f', [ -1.] )
b_b_lep_E_fitted    = array.array( 'f', [ -1.] )
b_b_lep_m_fitted    = array.array( 'f', [ -1.] )
b_b_lep_pt_fitted   = array.array( 'f', [ -1.] )
b_b_lep_y_fitted    = array.array( 'f', [ -1.] )
b_b_lep_phi_fitted  = array.array( 'f', [ -1.] )
b_t_lep_px_fitted   = array.array( 'f', [ -1.] )
b_t_lep_py_fitted   = array.array( 'f', [ -1.] )
b_t_lep_pz_fitted   = array.array( 'f', [ -1.] )
b_t_lep_E_fitted    = array.array( 'f', [ -1.] )
b_t_lep_m_fitted    = array.array( 'f', [ -1.] )
b_t_lep_pt_fitted   = array.array( 'f', [ -1.] )
b_t_lep_y_fitted    = array.array( 'f', [ -1.] )
b_t_lep_phi_fitted  = array.array( 'f', [ -1.] )

b_runNumber  = array.array( 'i', [ -1] )

tree = TTree( "nominal", "nominal" )
tree.Branch( 'runNumber',       b_runNumber,       'runNumber/i' )

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
tree.Branch( 'jetmu_pz_obs', b_jetmu_pz_obs, 'jetmu_pz_obs/F')
tree.Branch( 'jetmu_T0_obs', b_jetmu_T0_obs, 'jetmu_T0_obs/F')
tree.Branch( 'jetlep_ET_obs', b_jetlep_ET_obs, 'jetlep_ET_obs/F')
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

for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    

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

    jet_1_vect = MakeP4(jet_1[i], jet_1[i][4], representation)
    jet_2_vect = MakeP4(jet_2[i], jet_2[i][4], representation)
    jet_3_vect = MakeP4(jet_3[i], jet_3[i][4], representation)
    jet_4_vect = MakeP4(jet_4[i], jet_4[i][4], representation)
    jet_5_vect = MakeP4(jet_5[i], jet_5[i][4], representation)

    b_runNumber[0]  = i
    
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

tree.Print()
ofile.Write ()
ofile.Close ()
