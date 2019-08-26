import numpy as np
from ROOT import *
import itertools
from AngryTops.features import *
import pickle

# Mass and mass uncertainties for W and t in GeV
mW = 80.4
sigmaW = 20
mT = 172.5
sigmaT = 30

def MakeNeutrino(lep_Px, lep_Py, lep_Pz, met, met_phi):
    """
    Constructs the neutrino vector given the lepton momenta and the missing
    transverse energy and missing azimuthal energy
    """
    # v refers to the the neutrino
    v_pT = met
    v_px = met * np.cos(met_phi)
    v_py = met * np.sin(met_phi)

    # lepton already made by reco or particle levels:
    lep = TLorentzVector()
    lep.SetPxPyPzE(lep_Px, lep_Py, lep_Pz, 0)
    l_px = lep.Px()
    l_py = lep.Py()
    l_pz = lep.Pz()
    l_m  = lep.M()
    l_E  = lep.E()

    mdiff = 0.5 * ( mW*mW - l_m*l_m )
    pT_vl = v_px*l_px + v_py*l_py

    # Factors of the quadratic equation
    a = l_E*l_E - l_pz*l_pz
    b = -2. * l_pz * ( mdiff + pT_vl )
    c = v_pT*v_pT*l_E*l_E - mdiff*mdiff - pT_vl*pT_vl - 2.*mdiff*pT_vl

    # The discriminant
    delta = b*b - 4.*a*c

    if delta <= 0.:
        v_pz = -0.5 * b/a
    else:
        v_pz_1 = 0.5 * ( -b - np.sqrt(delta) ) / a
        v_pz_2 = 0.5 * ( -b + np.sqrt(delta) ) / a
        if np.abs(v_pz_1) < np.abs(v_pz_2):
            v_pz = v_pz_1
        else:
	        v_pz = v_pz_2

    v_E  = np.sqrt( v_pT*v_pT + v_pz*v_pz )

    m_neutrino = TLorentzVector()
    m_neutrino.SetPxPyPzE( v_px, v_py, v_pz, v_E )

    return m_neutrino

def ChiSquared(jets, nu, lep):
    """
    Compute the Chi-Squared Value for a predicted event.
    Jets is a list of 4 TLorentz vectors
    The order of the list elements:
    b_had, b_lep, W_had_jet1, W_had_jet2
    """
    W_had =  jets[2] + jets[3]
    W_lep = lep + nu
    t_had = W_had + jets[0]
    t_lep = W_lep + jets[1]

    chi2 =  (W_had.M() - mW)**2 / sigmaW**2 + (W_lep.M() - mW)**2 / sigmaW**2 + (t_had.M() - mT)**2 / sigmaT**2 + (t_lep.M() - mT)**2 / sigmaT**2

    return chi2, [jets[0], jets[1], W_had, W_lep, t_had, t_lep]


def reconstruct(jets, nu, lep):
    """Return (6 x 4) matrix with the following jet columns
    (b_had_px, b_lep_px, W_had_j1_px, W_had_j2_px)
    (b_had_py, b_lep_py, W_had_j1_py, W_had_j2_px)
    (b_had_pz, b_lep_pz, W_had_j1_pz, W_had_j2_px)
    (b_had_E,  b_lep_E,  W_had_j1_E,  W_had_j2_px)
    """
    # If only four jet event, remove the additional jet
    if np.all(jets[-1] == 0):
        jets = jets[:-1]
    combos = list(itertools.combinations(jets, 4))

    # These values will be updated in the end
    best_chi_squared = 10e9
    best_combo = combos[0]
    best_output = None

    # Go through all permutations of each combination and pick the best
    for i in range(len(combos)):
        combo = combos[i]
        permutes = list(itertools.permutations(combo))
        for j in range(len(permutes)):
            permute = permutes[i]
            chi_squared = ChiSquared(permute, nu, lep)
            if chi_squared < best_chi_squared:
                best_chi_squared, best_output = chi_squared
                best_combo = permute

    return best_combo, best_chi_squared


def FormatOutput(particles):
    """
    Format list of TLorentz vectors into a (2 x 3) matrix
    """
    t_had = particles[-2]
    t_lep = particles[-1]

    # Fill Output
    output = np.zeros(shape=(2, 3))
    output[0,0] += t_had.Px()
    output[0,1] += t_had.Py()
    output[0,2] += t_had.Pz()
    output[1,0] += t_lep.Px()
    output[1,1] += t_lep.Py()
    output[1,2] += t_lep.Pz()

    return output

def Predict(lep_arr, jet_arr):
    """
    Produced (2 x 3) output array given (6 x 6) input array.
    Calls helper functions in the following order:
    MakeNeutrino -> reconstruct -> FormatOutput
    """
    # Construct lepton
    lep = TLorentzVector()
    l_px = lep_arr[0]
    l_py = lep_arr[1]
    l_pz = lep_arr[2]
    l_E = np.sqrt(l_px**2 + l_py**2 + l_pz**2)
    l_T = lep_arr[3]  # We will remove this in later iterations
    met_pt = lep_arr[4]
    met_phi = lep_arr[5]
    lep.SetPxPyPzE(l_pz, l_py, l_pz, l_E)
    # Skip the fourth element which is the lepton arival time of flight
    nu = MakeNeutrino(lep_arr[0], lep_arr[1], lep_arr[2], lep_arr[4], lep_arr[5])

    jets = []
    for i in range(len(jet_arr)):
        j = TLorentzVector()
        j.SetPxPyPzE(jet_arr[i,0], jet_arr[i,1],jet_arr[i,2], jet_arr[i,3])
        jets.append(j)

    best_combo, best_chi_squared = reconstruct(jets, nu, lep)
    return FormatOutput(best_combo)

if __name__=="__main__":
    # Fixed variables dependant on my pipeline
    training_dir = "../CheckPoints/t_part_36_input"
    rep = "experimental"

    # Load Predictions and Scalars
    print("Loading predictions and scalars")
    predictions = np.load('{}/predictions.npz'.format(training_dir))
    testing_input = predictions['input']
    true = predictions['true']
    events = predictions['events']
    scaler_filename = "{}/scalers.pkl".format(training_dir)
    with open( scaler_filename, "rb" ) as file_scaler:
        jets_scalar = pickle.load(file_scaler)
        lep_scalar = pickle.load(file_scaler)
        output_scalar = pickle.load(file_scaler)

    # Rescale the truth array
    print("Rescaling inputs and outputs")
    true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
    true = output_scalar.inverse_transform(true)
    true = true.reshape(true.shape[0], -1, 3)

    # UNDO Scaling
    testing_input = testing_input.reshape(true.shape[0], 6, 6)
    lep = lep_scalar.inverse_transform(testing_input[:,0])
    jets = testing_input[:,1:,:-1]
    jets = jets.reshape(-1, 25)
    jets = jets_scalar.inverse_transform(jets)
    jets = jets.reshape(-1, 5, 5)
    testing_input = testing_input.reshape(true.shape[0], 36)

    # Predict each event
    print("Making predictions")
    chi2_pred = np.zeros(shape=(jets.shape[0], 2, 3))
    for i in range(jets.shape[0]):
        chi2_pred[i] += Predict(lep[i], jets[i])
        if i < 10:
            print(chi2_pred[i])

    # Save the predictions
    print("Saving predictions")
    np.savez("../CheckPoints/Chi2Model/predictions.npz" % training_dir,
                input=testing_input, true=true, pred=chi2_pred, events=events)

    # INPUT BREAKDOWN
    # When I actually load in the array, it is flattened to make things easier
    # but the first layer in most networks reshapes it immediately to a (6 x 6)
    # array
    # T = Arival Time of Flight (No idea what this means actually)
    # B = Btag Value
    # (6 x 6 array)
    # lep Px    j0 Px   j1 Px   j2 Px   j3 Px   j4 Px
    # lep Py    j0 Py   j1 Py   j2 Py   j3 Py   j4 Py
    # lep Pz    j0 Pz   j1 Pz   j2 Pz   j3 Pz   j4 Pz
    # lep T     j0 E    j1 E    j2 E    j3 E    j4 E
    # met       j0 M    j1 M    j2 M    j3 M    j4 M
    # met_phi   j0 B    j1 B    j2 B    j3 B    j4 B
