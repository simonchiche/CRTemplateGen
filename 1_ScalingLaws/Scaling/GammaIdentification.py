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
#endregion

#region Path definition
SimDir = "FAERIEParam" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "EnergyScaling" #"DepthScaling" #"ThetaScaling"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/MassComposition_backup/Protons"
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims_protons = []
Eradice_allsims_protons = []
Eradtot_protons = []
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


    ExC_proton, EyC_proton, EzC_proton, EtotC_proton = Shower.GetPeakTraces(Traces_C)
    ExG_proton, EyG_proton, EzG_proton, EtotG_proton = Shower.GetPeakTraces(Traces_G)

    #sys.exit(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradair_allsims_protons.append(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradice_allsims_protons.append(Shower.GetRadiationEnergyGeneric(Traces_G))
    Eradtot_protons.append(Shower.GetRadiationEnergyGeneric(Traces_C))

Eradair_allsims_protons = np.concatenate(Eradair_allsims_protons, axis =0)
Eradice_allsims_protons = np.concatenate(Eradice_allsims_protons, axis =0)


simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/MassComposition_backup/Gammas"
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims_gammas = []
Eradice_allsims_gammas = []
Eradtot_gammas = []
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
        ExC_gamma, EyC_gamma, EzC_gamma, EtotC_gamma = Shower.GetPeakTraces(Traces_C)
    ExG_gamma, EyG_gamma, EzG_gamma, EtotG_gamma = Shower.GetPeakTraces(Traces_G)

    #sys.exit(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradair_allsims_gammas.append(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradice_allsims_gammas.append(Shower.GetRadiationEnergyGeneric(Traces_G))
    Eradtot_gammas.append(Shower.GetRadiationEnergyGeneric(Traces_C))

Eradair_allsims_gammas = np.concatenate(Eradair_allsims_gammas, axis =0)
Eradice_allsims_gammas = np.concatenate(Eradice_allsims_gammas, axis =0)
# =============================================================================
#                             Plots
# =============================================================================
bins = np.linspace(0,300, 20)
plt.hist(EtotG_gamma, alpha=0.5, bins=bins)
plt.hist(EtotG_proton, alpha=0.5, bins=bins)



print(Eradair_allsims_gammas[:,4])

seldepth = 3116
#sel = (Eradair_allsims_gammas[:4] == seldepth)
sel = (Eradice_allsims_gammas[:,4] == 3116)  & (Eradice_allsims_gammas[:,5] ==0.0316)
Eradice_allsims_gammas_seltot = Eradice_allsims_gammas[sel][:,3]

sel = (Eradair_allsims_gammas[:,4] == 3116)  & (Eradair_allsims_gammas[:,5] ==0.0316)
Eradair_allsims_gammas_seltot = Eradair_allsims_gammas[sel][:,3]
zenithbins_gammas = Eradice_allsims_gammas[sel][:,6]

plt.scatter(zenithbins_gammas, Eradice_allsims_gammas_seltot)
plt.scatter(zenithbins_gammas, Eradair_allsims_gammas_seltot)
plt.yscale("log")
plt.show()



seldepth = 3116
#sel = (Eradair_allsims_gammas[:4] == seldepth)
sel = (Eradice_allsims_protons[:,4] == 3116)  & (Eradice_allsims_protons[:,5] ==0.0316)
Eradice_allsims_protons_seltot = Eradice_allsims_protons[sel][:,3]

sel = (Eradair_allsims_protons[:,4] == 3116)  & (Eradair_allsims_protons[:,5] ==0.0316)
Eradair_allsims_protons_seltot = Eradair_allsims_protons[sel][:,3]
zenithbins_protons = Eradice_allsims_protons[sel][:,6]

plt.scatter(zenithbins_protons, Eradice_allsims_protons_seltot)
plt.scatter(zenithbins_protons, Eradair_allsims_protons_seltot)
plt.show()


plt.scatter(zenithbins_protons, Eradice_allsims_protons_seltot/Eradair_allsims_protons_seltot)
plt.scatter(zenithbins_gammas, Eradice_allsims_gammas_seltot/Eradair_allsims_gammas_seltot)
plt.yscale("log")

arg =np.argsort(Eradice_allsims_gammas_seltot)
Eradice_allsims_gammas_seltot=Eradice_allsims_gammas_seltot[arg]
zenithbins_gammas= zenithbins_gammas[arg]
plt.errorbar(zenithbins_protons, Eradice_allsims_protons_seltot, yerr=0.25*Eradice_allsims_protons_seltot, label ="protons")
plt.errorbar(zenithbins_gammas, Eradice_allsims_gammas_seltot, yerr=0.25*Eradice_allsims_gammas_seltot, label ="gammas")
plt.yscale("log")
plt.xlabel("Zenith [Deg.]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice} [MeV]$")
plt.show()



fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=150)

ax.errorbar(
    zenithbins_protons,
    Eradice_allsims_protons_seltot,
    yerr=0.25 * Eradice_allsims_protons_seltot,
    fmt="o",
    markersize=5,
    capsize=4,
    elinewidth=1.5,
    label="Protons",
    color="#0072B2" 
)

ax.errorbar(
    zenithbins_gammas,
    Eradice_allsims_gammas_seltot,
    yerr=0.25 * Eradice_allsims_gammas_seltot,
    fmt="s",
    markersize=5,
    capsize=4,
    elinewidth=1.5,
    label=r"Gammas",
    color="#D55E00"
)

ax.set_yscale("log")

ax.set_xlabel(r"Zenith angle $\theta$ [Deg.]", fontsize=15)
ax.set_ylabel(r"$E_{\rm rad}^{\rm ice}$ [MeV]", fontsize=15)

ax.tick_params(axis="both", which="major", labelsize=13, direction="in", length=6)
ax.tick_params(axis="both", which="minor", direction="in", length=3)

ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.15)

ax.legend(fontsize=13, frameon=True)
plt.title(r"$E=10^{16.5}\, \rm eV$", fontsize=14)
plt.savefig("Mass_composition_ice_detectors.pdf", bbox_inches="tight")
plt.show()