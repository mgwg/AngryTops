import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.features import *
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

################################################################################
# CONSTANTS
inputdir = sys.argv[1]
outputdir = sys.argv[2]
representation = sys.argv[3]
subdir = '/jet_filter_bW_id_mass/'
if len(sys.argv) > 4:
    subdir = '/jet_filter_bW_id_mass{}/'.format(sys.argv[4])
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95
W_had_m_cutoff = (30, 130)

ALL = 0
NONE = 1
ONLY = 2
b_tagging = NONE    # 0/All: Consider all jets, both b-tagged and not b-tagged
                    # 1/None: Do not consider any b-tagged jets.
                    # 2/Only: Consider only b-tagged jets

np.set_printoptions(precision=3, suppress=True, linewidth=250)
model_filename  = "{}/simple_model.h5".format(inputdir)

################################################################################
# load data
print("INFO: fitting ttbar decay chain...")
predictions = np.load(inputdir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']
event_info = predictions['events']

particles_shape = (true.shape[1], true.shape[2])
print("jets shape", jets.shape)
print("b tagging option", b_tagging)
if scaling:
    scaler_filename = inputdir + "scalers.pkl"
    with open( scaler_filename, "rb" ) as file_scaler:
        jets_scalar = pickle.load(file_scaler)
        lep_scalar = pickle.load(file_scaler)
        output_scalar = pickle.load(file_scaler)
        # Rescale the truth array
        true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
        true = output_scalar.inverse_transform(true)
        true = true.reshape(true.shape[0], particles_shape[0], particles_shape[1])
        # Rescale the fitted array
        fitted = fitted.reshape(fitted.shape[0], fitted.shape[1]*fitted.shape[2])
        fitted = output_scalar.inverse_transform(fitted)
        fitted = fitted.reshape(fitted.shape[0], particles_shape[0], particles_shape[1])
        # Rescale the jets array
        jets_lep = jets[:,:6]
        jets_jets = jets[:,6:] # remove muon column
        jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) # reshape to 5 x 6 array

        # Remove the b-tagging states and put them into a new array to be re-appended later.
        b_tags = jets_jets[:,:,5]
        jets_jets = np.delete(jets_jets, 5, 2) # delete the b-tagging states

        jets_jets = jets_jets.reshape((jets_jets.shape[0], 25)) # reshape into 25 element long array
        jets_lep = lep_scalar.inverse_transform(jets_lep)
        jets_jets = jets_scalar.inverse_transform(jets_jets) # scale values ... ?
        #I think this is the final 6x6 array the arxiv paper was talking about - 5 x 5 array containing jets (1 per row) and corresponding px, py, pz, E, m
        jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))
        # Re-append the b-tagging states as a column at the end of jets_jets 
        jets_jets = np.append(jets_jets, np.expand_dims(b_tags, 2), 2)

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
# MAKE HISTOGRAMS

hists = {}

# Hadronic W
# True vs. obs
hists['had_W_dist'] = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_W_dist'].SetTitle("W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Jet matching invariant mass distributions
hists['had_W_obs_mass'] = TH1F("W_had_m","W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_W_obs_mass'].SetTitle("W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
# Jet matching transverse momentum distributions
hists['had_W_pT_diff'] = TH1F("h_pT_W_had_diff","W Hadronic p_{T} diffs, True - Observed", 50, -500, 500)
hists['had_W_pT_diff'].SetTitle("W Hadronic p_{T} diffs, True - Observed; Hadronic (GeV); A.U.")

# Hadronic b

hists['b_obs_mass'] = TH1F("b_m","b Quark Invariant Mass, Observed", 50, 0., 50. )
hists['b_obs_mass'].SetTitle("b Quark Invariant Mass, Observed; (GeV); A.U.")

# hists['had_b_pT_diff'] = TH1F("h_pT_b_had","b Quark p_{T}, Observed", 50, 0, 500)
# hists['had_b_pT_diff'].SetTitle("b Quark p_{T}, Observed; (GeV); A.U.")

# hists['had_b_dist'] = TH1F("h_b_true","b Quark Distances, True vs Observed", 50, 0, 3)
# hists['had_b_dist'].SetTitle("b Quark #eta-#phi distances, True vs Observed; (radians);A.U.")

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

    b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
    b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b , representation)

    b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
    b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b, representation)

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
    b_had_mass_diff = b_lep_mass_diff = W_had_mass_diff = 1000000

    # Perform jet matching for the bs, all jets, b-tagged and not b-tagged should be considered.
    for k in range(len(jets)): # loop through each jet to find the minimum distance for each particle
        b_had_d = m_b - jets[k].M()
        if b_had_d < b_had_mass_diff:
            b_had_mass_diff = b_had_d
            closest_b_had = jets[k]

    # match to b-quark twice for the hadronic and leptonic b
    for k in range(len(jets)): 
        b_lep_d = m_b - jets[k].M()
        # skip the jet if it's already matched to the hadronic b
        # leptonic b will always be less well matched because of this
        if (b_lep_d < b_lep_mass_diff) and (jets[k] != closest_b_had):
            b_lep_mass_diff = b_lep_d
            closest_b_lep = jets[k]

    good_jets = jets[:]
    if (b_tagging > 0):
        for m in range(len(jets)):
            # if don't include any b tagged jets and jet is b tagged OR
            # if only considering b tagged jets and jet is not b tagged
            if (b_tagging == 1 and jet_list[m, i, 5]) or (b_tagging == 2 and not jet_list[m,i,5]):
                good_jets.remove(jets[m])
    # If there are no jets remaining in good_jets, then skip this event. Don't populate histograms.
    if (not good_jets) or (len(good_jets) < 2):
        continue
    
    good_jets_pT = []
    for jet in good_jets:
        good_jets_pT.append(jet.Pt())
    if max(good_jets_pT) <= 50:
        continue

    # Consider best two jets, no cutoff so ignore that for now
    if (len(good_jets) >= 2):
        for k in range(len(good_jets)):
            # if good_jets only contains one element, loop is skipped since range would be (1,1)
            for j in range(k + 1, len(good_jets)):                  
                sum_vect = good_jets[k] + good_jets[j] 
                W_had_d_true = m_W - sum_vect.M()
                if W_had_d_true < W_had_mass_diff:
                    W_had_mass_diff = W_had_d_true
                    closest_W_had = sum_vect

        W_had_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
        W_had_dist = find_dist(W_had_true, sum_vect)
    
    # b quark calculations
    b_had_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
    b_lep_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()

    b_had_dist = find_dist(b_had_true, closest_b_had)
    b_lep_dist = find_dist(b_lep_true, closest_b_lep)

    # Fill hists

    if True: #(closest_W_had.M() >= W_had_m_cutoff[0] \
        #and closest_W_had.M() <= W_had_m_cutoff[1]):
        # b quarks
        hists['b_obs_mass'].Fill(closest_b_had.M())
        hists['b_obs_mass'].Fill(closest_b_lep.M())

        # don't consider dist and pT, because can't tell based on mass which one is the hadronic and which is the leptonic b
        # hists['had_b_dist'].Fill(np.float(closest_b_had.M()))
        # hists['had_b_pT_diff'].Fill(b_had_pT_diff)

        # Hadronic W
        hists['had_W_dist'].Fill(np.float(W_had_dist))
        hists['had_W_obs_mass'].Fill(closest_W_had.M())
        hists['had_W_pT_diff'].Fill(np.float(W_had_pT_diff))

try:
    os.mkdir('{0}'.format(outputdir + subdir))
except Exception as e:
    print("Overwriting existing files")

for key in hists:
    if 'corr' not in key:
        hist = hists[key]
        plot_hists(key, hist, outputdir+subdir)