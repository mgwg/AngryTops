import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists
import matplotlib as plt 

ALL = 0
NONE = 1
ONLY = 2

outputdir = sys.argv[1]
representation = sys.argv[2]
date = ''
if len(sys.argv) > 3:
    date = sys.argv[3]

subdir = '/jet_matching{}/'.format(date)
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
b_tagging = NONE    # 0/All: Consider all jets, both b-tagged and not b-tagged
                    # 1/None: Do not consider any b-tagged jets.
                    # 2/Only: Consider only b-tagged jets

# Cut ranges for the partons
W_had_m_cutoff = [25, 100]
W_had_pT_cutoff = [-50, 50]
W_had_dist_cutoff = [0, 0.7]

W_lep_ET_cutoff =[-100, 100]
W_lep_dist_cutoff = [0, 1]

b_had_pT_cutoff = [-60, 60]
b_had_dist_cutoff = [0, 0.7]

b_lep_pT_cutoff = [-60, 60]
b_lep_dist_cutoff = [0, 0.7]

# load data
predictions = np.load(outputdir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']

particles_shape = (true.shape[1], true.shape[2])

if scaling:
    scaler_filename = outputdir + "scalers.pkl"
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
    jets_jets = jets_jets.reshape((jets_jets.shape[0], 25))
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))

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

# Lists to hold percentages of events that pass cuts for each tolerance.
mass_percentages = []
pT_percentages = []
dist_percentages = []
# indices to increment maximum mass cutoff as (start, stop, step)
mass_range = range(W_had_m_cutoff[1], 155, 5)

# Function to analye each event and output whether or not it passes the required cuts.
def match_jets(i):
            
    W_had_true   = MakeP4( y_true_W_had[i], m_W, representation)
    W_lep_true   = MakeP4( y_true_W_lep[i], m_W , representation)

    b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
    b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)

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

    ################################################# true vs observed ################################################# 

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
        return (False, False, False)
    
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
        jet_combo_index = 1
    
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
        jet_combo_index = 2

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
        jet_combo_index = 0

            # # Special calculations for the observed leptonic W  
            # muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]]
            # nu_pT_obs = [jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] # Observed neutrino transverse momentum from missing energy as [x, y].
            # W_lep_Px_observed = muon_pT_obs[0] + nu_pT_obs[0]
            # W_lep_Py_observed = muon_pT_obs[1] + nu_pT_obs[1]
            
            # # Calculate the distance between true and observed phi.
            # lep_phi = np.arctan2(W_lep_Py_observed, W_lep_Px_observed)
            # W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
            # # Calculate transverse energy assuming daughter particles are massless
            # W_lep_ET_observed = np.sqrt( W_lep_Px_observed**2 + W_lep_Py_observed**2)
            # W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed
            # # Calculate the transverse mass
            # obs_daughter_angle = np.arccos( np.dot(muon_pT_obs, nu_pT_obs) / norm(muon_pT_obs) / norm(nu_pT_obs) )
            # met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(obs_daughter_angle))) 

            # # b quark calculations
            # b_had_true_obs_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
            # b_lep_true_obs_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()

    ############################################## check whether each event passes cuts #################################################
    # counter for hadronic W
    # Update tally for which jet combination is the closest
    W_had_pT_cut = W_had_dist_cut = False 

    W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
    if W_had_m_cut:
        W_had_pT_cut = (W_had_true_obs_pT_diff >= W_had_pT_cutoff[0] and W_had_true_obs_pT_diff <= W_had_pT_cutoff[1])
        W_had_dist_cut = (W_had_dist_true <= W_had_dist_cutoff[1]) 

        # # counter for lep W
        # W_lep_ET_cut = (W_lep_ET_diff >= W_lep_ET_cutoff[0] and W_lep_ET_diff <= W_lep_ET_cutoff[1])
        # W_lep_dist_cut = (W_lep_dist_true <= W_lep_dist_cutoff[1]) 
        # good_W_lep = (W_lep_ET_cut and W_lep_dist_cut)

        # W_lep_total_cuts += good_W_lep
        # W_lep_ET_cuts += W_lep_ET_cut
        # W_lep_dist_cuts += W_lep_dist_cut

        # # counter for hadronic b
        # b_had_pT_cut = (b_had_true_obs_pT_diff >= b_had_pT_cutoff[0] and b_had_true_obs_pT_diff <= b_had_pT_cutoff[1])
        # b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
        # good_b_had = (b_had_pT_cut and b_had_dist_cut)

        # b_had_total_cuts += good_b_had
        # b_had_pT_cuts += b_had_pT_cut
        # b_had_dist_cuts += b_had_dist_cut

        # # counter for leptonic b
        # b_lep_pT_cut = (b_lep_true_obs_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_obs_pT_diff <= b_lep_pT_cutoff[1])
        # b_lep_dist_cut = (b_lep_dist_true <= b_lep_dist_cutoff[1]) 
        # good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

        # b_lep_total_cuts += good_b_lep
        # b_lep_pT_cuts += b_lep_pT_cut
        # b_lep_dist_cuts += b_lep_dist_cut

        # # Good events must pass cuts on all partons.
        # good_event += (good_b_had and good_b_lep and good_W_had and good_W_lep)
    return (W_had_m_cut, W_had_pT_cut, W_had_dist_cut)


# Function to make histograms
def calc_percentages():

    for i in mass_range:
        W_had_m_cutoff[1] = i

        # Counters to make tally number of events that pass cuts
        W_had_m_total = 0.
        W_had_pT_total = 0.
        W_had_dist_total = 0.

            # W_lep_total_cuts = 0.
            # W_lep_ET_cuts = 0.
            # W_lep_dist_cuts = 0.

            # b_had_pT_cuts = 0.
            # b_had_dist_cuts = 0.
            # b_had_total_cuts = 0.

            # b_lep_pT_cuts = 0.
            # b_lep_dist_cuts = 0.
            # b_lep_total_cuts = 0.

            # good_event = 0.

        # loop through every event
        for j in range(n_events): 
            if j == 0 :
                print("Mass cutoff: {}".format(i))
            (W_had_m_cut, W_had_pT_cut, W_had_dist_cut) = match_jets(j)
            
            W_had_m_total += W_had_m_cut
            W_had_pT_total += W_had_pT_cut
            W_had_dist_total += W_had_dist_cut

        # calculate percentage of good events
        mass_perc = 100.*W_had_m_total/n_events
        pT_perc = 100.*W_had_pT_total/n_events
        dist_perc = 100.*W_had_dist_total/n_events

        mass_percentages.append(mass_perc)
        pT_percentages.append(pT_perc)
        dist_percentages.append(dist_perc)



def plot():
    plt.plot(history.epoch, mass_percentages, linestyle = 'solid', color = 'C0', label='Mass')
    plt.plot(mass_range, pT_percentages, marker = "+", color = 'C9', label='Events that pass pT cut of {}'.format(W_had_pT_cutoff))
    plt.plot(mass_range, dist_percentages, marker = "v", color = 'C4', label=r'Events that pass $\eta - \phi$ cut of {}'.format(W_had_dist_cutoff))

    plt.xlabel('Maximum Mass Cutoff [GeV]')
    plt.ylabel('%')
    plt.legend()

    plt.xlim([0, mass_range[-1]])
    plt.savefig("{}/mass_cuts.png".format(subdir))
    plt.clf()
    return True

# Run the two helper functions above   
if __name__ == "__main__":
    try:
        os.mkdir('{}/{}'.format(outputdir, subdir))
    except Exception as e:
        print("Overwriting existing files")
    print("jets shape", jets.shape)
    print("b tagging option", b_tagging)
    calc_percentages() 
    plot()