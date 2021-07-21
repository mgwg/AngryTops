import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

ALL = 0
NONE = 1
ONLY = 2

outputdir = sys.argv[1]
representation = sys.argv[2]
name = 'mass_cuts'
if len(sys.argv) > 3:
    name = sys.argv[3]

subdir = '/jet_matching/'
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
b_tagging = NONE    # 0/All: Consider all jets, both b-tagged and not b-tagged
                    # 1/None: Do not consider any b-tagged jets.
                    # 2/Only: Consider only b-tagged jets

# Cut ranges for the partons
W_had_m_cutoff = [25, 100]
W_had_pT_cutoff = [-20, 20]
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
mass_range = range(W_had_m_cutoff[1], 180, 5) #125, 5)
pT_range = range(W_had_pT_cutoff[1], 100, 5) #40, 5)
mass_range, pT_range = np.meshgrid(mass_range, pT_range)
total_percentages = np.zeros(mass_range.shape)

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
    W_had_dist_true = 1000000

    good_jets = jets[:]
    if (b_tagging > 0):
        for m in range(len(jets)):
            # if don't include any b tagged jets and jet is b tagged OR
            # if only considering b tagged jets and jet is not b tagged
            if (b_tagging == 1 and jet_list[m, i, 5]) or (b_tagging == 2 and not jet_list[m,i,5]):
                good_jets.remove(jets[m])
    # If there are no jets remaining in good_jets, then skip this event. Don't populate histograms.
    if not good_jets:
        return False
    
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

    ############################################## check whether each event passes cuts #################################################
    # counter for hadronic W
    # Update tally for which jet combination is the closest

    W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
    W_had_pT_cut = (W_had_true_obs_pT_diff >= W_had_pT_cutoff[0] and W_had_true_obs_pT_diff <= W_had_pT_cutoff[1])

    return W_had_m_cut and W_had_pT_cut


# Function to make histograms
def calc_percentages():

    rows = mass_range.shape[0]
    cols = mass_range.shape[1]

    for x in range(0, rows):
        print("{} out of {}".format(x, rows))
        for y in range(0, cols):
            # print("{} out of {}".format(y, cols))

            W_had_m_cutoff[1] = mass_range[x][y]
            W_had_pT_cutoff[1] = pT_range[x][y]
            W_had_pT_cutoff[0] = -1*pT_range[x][y]

            # Counters to make tally number of events that pass cuts
            W_had_total_cuts = 0

            # loop through every event
            for j in range(n_events): 
                good_event = match_jets(j)
                W_had_total_cuts += good_event

            # calculate percentage of good events
            total_perc = 100.*W_had_total_cuts/n_events

            total_percentages[x][y] = total_perc



def plot():

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mass_range, pT_range, total_percentages, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig("{}/{}/{}.png".format(outputdir, subdir, name))
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