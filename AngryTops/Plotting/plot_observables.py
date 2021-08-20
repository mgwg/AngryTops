#!/usr/bin/env python
import os, sys
from ROOT import *
import numpy as np
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *

# INPUT
training_dir = sys.argv[1]
caption = sys.argv[2]
if caption == "None": caption = None
if len(sys.argv) > 3:
    attributes = attributes_tquark
    corr_2d = corr_2d_tquark

gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

def plot_observables(obs):
    # Load the histograms
    hname_true = "%s_true" % (obs)
    hame_fitted = "%s_fitted" % (obs)

    # True and fitted leaf
    h_true = infile.Get(hname_true)
    h_fitted = infile.Get(hame_fitted)
    if h_true == None:
        print ("ERROR: invalid histogram for", obs)

    # Axis titles
    xtitle = h_true.GetXaxis().GetTitle()
    ytitle = h_true.GetYaxis().SetTitle("A.U.")
    if h_true.Class() == TH2F.Class():
        h_true = h_true.ProfileX("pfx")
        h_true.GetYaxis().SetTitle( ytitle )
    else:
        Normalize(h_true)
        Normalize(h_fitted)

    # Set Style
    SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_true.Draw("h")
    h_fitted.Draw("h same")
    hmax = 1.5 * max( [ h_true.GetMaximum(), h_fitted.GetMaximum() ] )
    h_fitted.SetMaximum( hmax )
    h_true.SetMaximum( hmax )
    h_fitted.SetMinimum( 0. )
    h_true.SetMinimum( 0. )

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_true, "MG5+Py8", "f" )
    leg.AddEntry( h_fitted, "Predicted", "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    KS = h_true.KolmogorovTest( h_fitted )
    X2 = ChiSquared(h_true, h_fitted) # UU NORM
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.7, 0.80, "KS test: %.2f" % KS )
    l.DrawLatex( 0.7, 0.75, "#chi^{2}/NDF = %.2f" % X2 )

    gPad.RedrawAxis()
    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

        gPad.RedrawAxis()

    pad1.cd()

    yrange = [0.4, 1.6]
    frame, tot_unc, ratio = DrawRatio(h_true, h_fitted, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/img/{1}.png".format(training_dir, obs))
    pad0.Close()
    pad1.Close()
    c.Close()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_residuals(obs):
    hist_name = "diff_{0}".format(obs)

    # True and fitted leaf
    hist = infile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", obs)

    #Normalize(hist)
    if hist.Class() == TH2F.Class():
        hist = hist.ProfileX("hist_pfx")

    fwhm, sigma = getFwhm( hist )

    SetTH1FStyle( hist,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=2, markersize=1 )

    c, pad0 = MakeCanvas2()

    pad0.cd()
    hist.GetXaxis().SetNdivisions(508)
    #hist.GetXaxis().SetLabelSize( 0.015 )
    #hist.GetXaxis().SetTitleSize(0.015)
    #hist.GetYaxis().SetLabelSize( 0.015 )
    hist.Draw()

    hmax = 1.25 * max( [ hist.GetMaximum(), hist.GetMaximum() ] )
    hmin = 1.25 * min([ hist.GetMaximum(), hist.GetMaximum() ])
    hist.SetMaximum(hmax)
    hist.SetMinimum(hmin)


    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.7, 0.80, "fwhm: %.2f" % fwhm )
    l.DrawLatex( 0.7, 0.75, "sigma: %.2f" % sigma )

    gPad.RedrawAxis()

    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

    c.cd()

    c.SaveAs("{0}/img/diff_{1}.png".format(training_dir,obs))
    pad0.Close()
    c.Close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_correlations(hist_name):

    # True and fitted leaf
    hist = infile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", hist_name)

    #Normalize(hist)

    SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
    #pad0.SetTopMargin( 0.14 )
    pad0.SetTopMargin( 0.07 ) #0.05
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()
    pad0.cd()

    hist.Draw("colz")

    corr = hist.GetCorrelationFactor()
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )

    gPad.RedrawAxis()

    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

    c.cd()

    c.SaveAs("{0}/img/{1}.png".format(training_dir, hist_name))
    pad0.Close()
    c.Close()


def plot_profile(obs):
    hist_name = "reso_{0}".format(obs)
    hist = infile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", obs)
    hist = hist.ProfileX("hist_pfx")
    SetTH1FStyle( hist,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=1 )
    c = TCanvas()

    hist.Draw()

    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

    c.cd()

    c.SaveAs("{0}/img/reso_{1}.png".format(training_dir,obs))
    c.Close()

################################################################################
if __name__==   "__main__":
    try:
        os.mkdir('{}/img'.format(training_dir))
    except Exception as e:
        print("Overwriting existing files")
    infilename = "{}/histograms.root".format(training_dir)
    infile = TFile.Open(infilename)

    # Make a plot for each observable
    for obs in attributes:
        plot_observables(obs)

    # Draw Differences and resonances
    for obs in attributes:
        plot_residuals(obs)
        plot_profile(obs)

    # Draw 2D Correlations
    for corr in corr_2d:
        plot_correlations(corr)
