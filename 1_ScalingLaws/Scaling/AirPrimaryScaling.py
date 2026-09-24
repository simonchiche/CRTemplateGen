#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""

# Modules import
#region Modules 
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
from MainModules.ShowerClass import CreateShowerfromHDF5
from MainModules.PlotConfig import MatplotlibConfig
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpoleVpoleEradRatiovsThetavsE, PlotEradtotThetaScaling, GetMeanEradScalingVsE, PlotMeanEradScalingVsE, PlotEradIceEScalingvsDepth, PlotGroundParticleEVsZenith, PlotEradIcevsZenE, PlotEradIcevsEgroundPart, EradicevsZenE, GetEradvsEgroundPart, GetGroundParticleEnergy, GetEgroundPart_E
from Modules.ModuleEnergyScaling import PlotEradEnergyScaling_zenbin, PlotEradEnergyScaling_normalized, PlotEradEnergyScaling_vs_theta
#endregion

#region Path definition
SimDir = "FAERIEParam" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "EnergyScaling" #"DepthScaling" #"ThetaScaling"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/FullDenseDeepCr"
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims = []
Eradice_allsims = []
Eradtot = []
sin_alpha_all = []
counter = 0
XmaxDistAll= dict()
Bgeo = np.array([7.705, 0, 54.111])
uB = Bgeo/np.linalg.norm(Bgeo)

for simpath in SimpathAll:

    print(simpath.split("/")[-1])
    Shower = CreateShowerfromHDF5(simpath)
    # =============================================================================
    #                              Load Traces
    # =============================================================================

    energy, theta, xmaxdist, Nant = Shower.energy, Shower.zenith, Shower.xmaxdist/1e5, Shower.nant
    XmaxDistAll[energy, theta]= xmaxdist
    print(xmaxdist)
    uv = np.array([np.sin(theta)*np.cos(0), np.sin(theta)*np.sin(0), -np.cos(theta)])
    sin_alpha = np.sin(np.arccos(np.dot(uv, uB)))
    sin_alpha_all.append(sin_alpha)

    Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos

    Nlay, Nplane, Depths = Shower.GetDepths()
    #Traces_tot = Shower.CombineTraces()

    # =============================================================================
    #                                Filter
    # =============================================================================

    Filter = True
    if(Filter):
        fs, lowcut, highcut = 5e9, 50e6, 1e9
        #Traces_C_filtered =Shower.filter_all_traces(Traces_C, fs, lowcut, highcut)
        #Traces_G_filered =Shower.filter_all_traces(Traces_G, fs, lowcut, highcut)

        Traces_C =Shower.filter_all_traces(Traces_C, fs, lowcut, highcut)
        Traces_G =Shower.filter_all_traces(Traces_G, fs, lowcut, highcut)

    # =============================================================================
    #                         Radiation energy
    # =============================================================================

    #sys.exit(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradair_allsims.append(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradice_allsims.append(Shower.GetRadiationEnergyGeneric(Traces_G))
    Eradtot.append(Shower.GetRadiationEnergyGeneric(Traces_C))

Eradair_allsims = np.concatenate(Eradair_allsims, axis =0)
Eradice_allsims = np.concatenate(Eradice_allsims, axis =0)

# =============================================================================
#                             Plots
# =============================================================================


### Energy Scaling ###
# In-air radiation energy vs primary energy
SelDepth = 3116
title = "In-air"

PlotEradEnergyScaling_zenbin(Eradair_allsims, SelDepth, title, Shower, OutputPath)

### TEST0: Erad/Ep^2 vs E
def model_linear(x, k):
    return k * x

def PlotEradEnergyScaling_normalized(Erad_allsims, SelDepth, title, Shower, OutputPath):

    ZenithAll = np.unique(Erad_allsims[:,6])

    for i in range(0,8,1):#len(ZenithAll)):
        sel = (Erad_allsims[:,6] == ZenithAll[i]) & (Erad_allsims[:,4] == SelDepth)

        arg = np.argsort(Erad_allsims[sel][:,5])
        EnergyBins= np.unique(Erad_allsims[sel][:,5])
        Erad_lin = np.sqrt(Erad_allsims[sel][:,3][arg])
        Erad_lin = Erad_allsims[sel][:,3][arg]

        popt, pcov = curve_fit(model_linear, EnergyBins, Erad_lin)

        k=popt[0]
        print(k)
        plt.scatter(EnergyBins**2, Erad_lin/Erad_lin[1], marker="s",\
                 label ="$\\theta =%.d^{\circ}$" %ZenithAll[i])
        

    #plt.plot(EnergyBins, k*EnergyBins/(k*EnergyBins[0]))
    data_fit = EnergyBins**2*0.1/(min(EnergyBins**2))
    fit_low  = data_fit * (1 - 0.15)
    fit_high = data_fit * (1 + 0.15)
    plt.plot(EnergyBins**2, data_fit, label = r"linear scaling", color="#D94A38")
    plt.fill_between(EnergyBins**2, fit_low, fit_high, color="#D94A38", alpha=0.2)
    plt.xlabel(r"[$\mathcal{E}_p/1\,\mathrm{EeV}]^{2}$")
    plt.ylabel(r"$E_{\rm rad}/[E_{\rm rad}(\mathcal{E}_p=10^{17}\, \mathrm{eV})]$")
    plt.legend()
    plt.yscale('log')
    plt.xscale('log') 
    plt.title(title + ", Depth$=%.d\,$m" %(Shower.glevel- SelDepth), fontsize=14)
    plt.legend(loc = "upper left", ncol=2, fontsize=11)
    plt.savefig(OutputPath + "_" + title + "_vs_E_|z|%.d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()
    return


#### All normalized by respective value at 10^17 eV
PlotEradEnergyScaling_normalized(Eradair_allsims, SelDepth, title, Shower, OutputPath)

### TEST3: Erad/Ep vs theta 
SelDepth = 3116
PlotEradEnergyScaling_vs_theta(Eradair_allsims, SelDepth, title, Shower, OutputPath)



