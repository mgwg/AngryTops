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
        fwhm = getFwhm(hist)
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

    return fwhm
