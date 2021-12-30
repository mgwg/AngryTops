from ROOT import *
import numpy as np

# Helper function to create histograms of eta-phi distance distributions
def MakeP4(y, m, representation):
    """
    Form the momentum vector.
    """
    p4 = TLorentzVector()
    p0 = y[0]
    p1 = y[1]
    p2 = y[2]
    # Construction of momentum vector depends on the representation of the input
    if representation == "pxpypzE":
        E  = y[3]
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "pxpypzM":
        M  = y[3]
        E = np.sqrt(p0**2 + p1**2 + p2**2 + M**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "pxpypz" or representation == "pxpypzEM":
        E = np.sqrt(p0**2 + p1**2 + p2**2 + m**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "ptetaphiE":
        E  = y[3]
        p4.SetPtEtaPhiE(p0, p1, p2, E)
    elif representation == "ptetaphiM":
        M  = y[3]
        p4.SetPtEtaPhiM(p0, p1, p2, M)
    elif representation == "ptetaphi" or representation == "ptetaphiEM":
        p4.SetPtEtaPhiM(p0, p1, p2, m)
    else:
        raise Exception("Invalid Representation Given: {}".format(representation))
    return p4

def find_dist(a, b):
    '''
    a, b are both TLorentz Vectors
    returns the eta-phi distances between true and sum_vect
    '''
    dphi_true = min(np.abs(a.Phi() - b.Phi()), 2*np.pi-np.abs(a.Phi() - b.Phi()))
    deta_true = a.Eta() - b.Eta()
    d_true = np.sqrt(dphi_true**2 + deta_true**2)
    return d_true

# Helper function to output and save the histograms and scatterplots 
def plot_hists(key, hist, outputdir):
    c1 = TCanvas()
    hist.Draw()

    # Make semi-log plots:
    if "semilog" in key:
        c1.SetLogy()

    # Make log-log plots:
    if "loglog" in key:
        c1.SetLogx()
        c1.SetLogy()

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    # Display bin width for all plots except scatterplots
    if "scat" not in key:
        binWidth = hist.GetBinWidth(0)
        legend.DrawLatex( 0.65, 0.70, "Bin Width: %.2f" % binWidth )
    if 'chi' in key:
        fwhm, sigma = getFwhm(hist)
        legend.DrawLatex( 0.65, 0.65, "FWHM: %.2f" % fwhm )

    c1.SaveAs(outputdir + key +'.png')
    c1.Close()

def getFwhm(hist):
    """
    Calculates the fwhm of hist.
    @Parameters:
    hist: TH1 Class
    """

    #fwhm
    halfMax = hist.GetMaximum()/2.0
    nMax = hist.GetMaximumBin()
    
    nbins = hist.GetNbinsX()
    bin1 = 0
    bin2 = nbins
    for i in range(1, nbins + 1):
        y = hist.GetBinContent(i)
        # find the greatest bin index left of max with bin content <= to half-max, and get its centre
        if (i < nMax and y <= halfMax) and (i > bin1):
            bin1 = hist.GetBinCenter(i)
        # find the smallest bin index right of max with bin content <= to half-max, and get its centre
        if (i > nMax and y <= halfMax) and (i < bin2):
            bin2 = hist.GetBinCenter(i)
    fwhm = bin2 - bin1
    sigma = fwhm/2.3548

    return fwhm, sigma

# Normalize a histogram
def Normalize( h, sf=1.0 ):
    if h == None: return
    # Calculate area of histogram
    Area = h.Integral()
    if Area == 0.: return
    # Multiply all bars of histogram by sf/Area, usually < 1.
    h.Scale( sf / Area, "nosw2") # Need "nosw2" option to retain correct histogram style.

def plot_corr(key, hist, outputdir):

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 )
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
    pad0.SetTopMargin( 0.07 )
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()
    pad0.cd()

    hist.Draw("colz")

    corr = hist.GetCorrelationFactor()
    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack)
    legend.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )

    gPad.RedrawAxis()

    caption = hist.GetName()
    newpad = TPad("newpad","a caption",0.1,0,1,1)
    newpad.SetFillStyle(4000)
    newpad.Draw()
    newpad.cd()
    title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
    title.SetFillColor(16)
    title.SetTextFont(52)
    if 'pT' or 'ET' in key:
        title.SetTextSize(0.8)
    title.Draw()

    c.cd()
    c.SaveAs(outputdir + key +'.png')
    pad0.Close()
    c.Close()

def undo_scaling(jets_scalar, lep_scalar, output_scalar, jets=[], true=[], fitted=[]):
    particles_shape = (true.shape[1], true.shape[2])
    if len(true):
        # Rescale the truth array
        true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
        true = output_scalar.inverse_transform(true)
        true = true.reshape(true.shape[0], particles_shape[0], particles_shape[1])
    if len(fitted):
        # Rescale the fitted array
        fitted = fitted.reshape(fitted.shape[0], fitted.shape[1]*fitted.shape[2])
        fitted = output_scalar.inverse_transform(fitted)
        fitted = fitted.reshape(fitted.shape[0], particles_shape[0], particles_shape[1])
    if len(jets):
        # Rescale the jets array
        jets_lep = jets[:,:6]
        jets_jets = jets[:,6:] # remove muon column
        jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) # reshape to 5 x 6 array
        # Remove the b-tagging states and put them into a new array to be re-appended later.
        b_tags = jets_jets[:,:,5]
        jets_jets = np.delete(jets_jets, 5, 2)

        jets_jets = jets_jets.reshape((jets_jets.shape[0], 25)) # reshape into 25 element long array
        jets_lep = lep_scalar.inverse_transform(jets_lep)
        jets_jets = jets_scalar.inverse_transform(jets_jets)
        # 5 x 5 array containing jets (1 per row) and corresponding px, py, pz, E, m
        jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))
        # Re-append the b-tagging states as a column at the end of jets_jets 
        jets_jets = np.append(jets_jets, np.expand_dims(b_tags, 2), 2)

    return jets_jets, jets_lep, true, fitted