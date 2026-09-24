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


# In-air radiation energy vs zenith angle
SelE = 0.316
title = "In-air"
SelZen =0
Eradair_allsims_corrected = Eradair_allsims[Eradair_allsims[:,5] == SelE]

Eradair_allsims_corrected[:,:4] = Eradair_allsims_corrected[:,:4]

PlotEradThetaScaling(Eradair_allsims_corrected, Depths, SelE, SelZen, title, OutputPath)

# In-ice radiation energy vs zenith angle
SelE = 0.316
title = "In-ice"
PlotEradThetaScaling(Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)



# Air, Ice radiation energy vs zenith angle
SelE = 0.0316
def PlotEradAirIceThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    Ex_depth, Ey_depth, Ez_depth = [], [], []
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{x,\mathrm{air}}$", color="#0072B2", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,1][arg], label ="$E^{\mathrm{rad}}_{y,\mathrm{air}}$", color="#E69F00", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,2][arg], label ="$E^{\mathrm{rad}}_{z,\mathrm{air}}$", color="#CC79A7", linestyle='dashed')
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{\mathrm{rad}} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend(ncol=2, framealpha=0.8)
        plt.grid(alpha=0.3)
        plt.title(" $E=10^{17.5}\,$eV, Depth =%d m" %(3216-Depths[i]), fontsize=14)
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

        Ex_depth.append(Eradair_allsims[sel][:,0][arg])
        Ey_depth.append(Eradair_allsims[sel][:,1][arg])
        Ez_depth.append(Eradair_allsims[sel][:,2][arg])

    return np.array(Eradair_allsims[sel][:,6][arg]), np.array(Ex_depth), np.array(Ey_depth), np.array(Ez_depth)
title ="AirIce"
zen_sorted, Ex0, Ey0, Ez0 = PlotEradAirIceThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)



def PlotEradAirThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    Ex_depth, Ey_depth, Ez_depth = [], [], []
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        arg = np.argsort(Eradair_allsims[sel][:,6])
        Eradx = Eradair_allsims[sel][:,0][arg]
        Erady = Eradair_allsims[sel][:,1][arg]
        Eradz = Eradair_allsims[sel][:,2][arg]
        Eradtot = Eradair_allsims[sel][:,3][arg]
        plt.plot(Eradair_allsims[sel][:,6][arg],Eradx, label ="$E^{\mathrm{rad}}_{x,\mathrm{air}}$", color="#0072B2", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Erady, label ="$E^{\mathrm{rad}}_{y,\mathrm{air}}$", color="#E69F00", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradz, label ="$E^{\mathrm{rad}}_{z,\mathrm{air}}$", color="#CC79A7", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradtot, label ="$E^{\mathrm{rad}}_{tot,\mathrm{air}}$", color="#009E73", linestyle='dashed')
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{\mathrm{rad}} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend(ncol=2, framealpha=0.8)
        plt.grid(alpha=0.3)
        plt.title(" $E=10^{17.5}\,$eV, Depth =%d m" %(3216-Depths[i]), fontsize=14)
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

        Ex_depth.append(Eradair_allsims[sel][:,0][arg])
        Ey_depth.append(Eradair_allsims[sel][:,1][arg])
        Ez_depth.append(Eradair_allsims[sel][:,2][arg])

    return np.array(Eradair_allsims[sel][:,6][arg]), np.array(Ex_depth), np.array(Ey_depth), np.array(Ez_depth)

zen_sorted, Ex0, Ey0, Ez0 = PlotEradAirThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)



def get_sin_alpha(theta):
    Bgeo = np.array([7.705, 0, -54.111])
    uB = Bgeo/np.linalg.norm(Bgeo)
    uv = np.array([np.sin(theta)*np.cos(0), 180, -np.cos(theta)])
    sin_alpha = np.sin(np.arccos(np.dot(uv, uB)))
    return sin_alpha



def _getAirDensity(_height, model):

    '''Returns the air density at a specific height, using either an 
    isothermal model or the Linsley atmoshperic model as in ZHAireS

    Parameters:
    ---------
        h: float
            height in meters

    Returns:
    -------
        rho: float
            air density in g/cm3
    '''

    if model == "isothermal":
            #Using isothermal Model
            rho_0 = 1.225*0.001    #kg/m^3
            M = 0.028966    #kg/mol
            g = 9.81        #m.s^-2
            T = 288.        #
            R = 8.32        #J/K/mol , J=kg m2/s2
            rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3

    elif model == "linsley":
        #Using Linsey's Model
        bl = np.array([1222., 1144., 1305.5948, 540.1778,1])*10  # g/cm2  ==> kg/cm3
        cl = np.array([9941.8638, 8781.5355, 6361.4304, 7721.7016, 1e7])  #m
        hl = np.array([4,10,40,100,113])*1e3  #m
        if (_height>=hl[-1]):  # no more air
            rho = 0
        else:
            hlinf = np.array([0, 4,10,40,100])*1e3  #m
            ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
            rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
            rho = rho[0]*0.001
    else:
        print("#### Error in GetDensity: model can only be isothermal or linsley.")
        return 0

    return rho

ZenithMaskedSorted, XmaxHeight16_5, XmaxHeight17, XmaxHeight17_5 = np.loadtxt("./Data/XmaxPos_XmaxHeightvsZenith.txt", unpack=True)


rho_xmax_16_5, rho_xmax_17, rho_xmax_17_5 = [],[],[]

for i in range(len(XmaxHeight16_5)):
    rho_xmax_16_5.append(_getAirDensity(XmaxHeight16_5[i], "linsley"))
    rho_xmax_17.append(_getAirDensity(XmaxHeight17[i], "linsley"))
    rho_xmax_17_5.append(_getAirDensity(XmaxHeight17_5[i], "linsley"))

rho_xmax_16_5 = np.array(rho_xmax_16_5)
rho_xmax_17 = np.array(rho_xmax_17)
rho_xmax_17_5 = np.array(rho_xmax_17_5)


def PlotEradAirThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath, rho_xmax_16_5, rho_xmax_17, rho_xmax_17_5):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    Ex_depth, Ey_depth, Ez_depth = [], [], []
    Ebins = np.unique(Eradair_allsims[:,5])
    SelDepth = 3216
    ZenithBins = np.unique(Eradair_allsims[:,6])
    print(ZenithBins)
    sin_alpha_all = []
    for zen in ZenithBins:
        sin_alpha_all.append(get_sin_alpha(zen*np.pi/180))
    sin_alpha_all = np.array(sin_alpha_all)
    print(sin_alpha_all)
    print(Ebins)
    Eradtot_all = dict()
    for i in range(len(Ebins)):

        sel = (Eradair_allsims[:,4] == SelDepth) & (Eradair_allsims[:,5] == Ebins[i])
        
        arg = np.argsort(Eradair_allsims[sel][:,6])
        Eradx = Eradair_allsims[sel][:,0][arg]
        Erady = Eradair_allsims[sel][:,1][arg]
        Eradz = Eradair_allsims[sel][:,2][arg]
        Eradtot = Eradair_allsims[sel][:,3][arg]
        plt.plot(Eradair_allsims[sel][:,6][arg],Eradx, label ="$E^{\mathrm{rad}}_{x,\mathrm{air}}$", color="#0072B2", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Erady, label ="$E^{\mathrm{rad}}_{y,\mathrm{air}}$", color="#E69F00", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradz, label ="$E^{\mathrm{rad}}_{z,\mathrm{air}}$", color="#CC79A7", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradtot, label ="$E^{\mathrm{rad}}_{tot,\mathrm{air}}$", color="#009E73", linestyle='dashed')
        #plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{\mathrm{rad}} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend(ncol=1, framealpha=0.8)
        plt.grid(alpha=0.3)
        plt.title(" $E=10^{17.5}\,$eV, Depth =%d m" %(SelDepth), fontsize=14)
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, SelDepth), bbox_inches = "tight")
        plt.show()

        Ex_depth.append(Eradair_allsims[sel][:,0][arg])
        Ey_depth.append(Eradair_allsims[sel][:,1][arg])
        Ez_depth.append(Eradair_allsims[sel][:,2][arg])
        Eradtot_all[i] = Eradtot/Ebins[i]**2

    ZenithBins = Eradair_allsims[sel][:,6][arg]
    plt.plot(ZenithBins, 1/sin_alpha_all**2)
    k = 1/sin_alpha_all**2
    plt.show()

    plt.plot(rho_xmax_16_5[2:]*1000, Eradtot_all[0][2:]*k[2:]*1.25, "-o", label =r"$E = %.2f\,$EeV" %Ebins[0])
    plt.plot(rho_xmax_17[2:]*1000, Eradtot_all[1][2:]*k[2:], "-", label =r"$E = %.2f\,$EeV" %Ebins[1])
    arg = np.argsort(rho_xmax_17_5[2:])
    plt.plot(rho_xmax_17_5[2:][arg]*1000, Eradtot_all[2][2:][arg]*k[2:][arg]*1.05, "-x", label =r"$E = %.2f\,$EeV" %Ebins[2])
    plt.xlabel("$\\rho_{xmax} \, $[kg/m$^3$]")
    plt.ylabel("$E_{rad,tot}/sin(alpha)**2 \, $[MeV]")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend(ncol=1, framealpha=0.8)
    plt.savefig(OutputPath + "_" + title + "_vs_Energy_zen%.1f_z%d.pdf" %(SelZen, SelDepth), bbox_inches = "tight")
    #plt.ylim(50, 500)
    plt.show()

    rhoXmaxall = np.concatenate((rho_xmax_16_5[0:]*1000, rho_xmax_17[2:]*1000, rho_xmax_17_5[2:][arg]*1000))
    Eradtotall = np.concatenate((Eradtot_all[0][0:]*k[0:]*1.25, Eradtot_all[1][2:]*k[2:], Eradtot_all[2][2:][arg]*k[2:][arg]*1.05))

    
    plt.scatter(rhoXmaxall, Eradtotall)
    plt.xlabel("$\\rho_{xmax} \, $[kg/m$^3$]")
    plt.ylabel("$E_{rad,tot}/sin(alpha)**2 \, $[MeV]")
    plt.show()

    #plt.plot(Eradair_allsims[sel][:,6][arg], Eradtot_all[0]*k, "-o", label =r"$E = %.2f\,$EeV" %Ebins[0])
    #plt.plot(Eradair_allsims[sel][:,6][arg], Eradtot_all[1]*k, "-", label =r"$E = %.2f\,$EeV" %Ebins[1])
    #plt.plot(Eradair_allsims[sel][:,6][arg], Eradtot_all[2]*k, "-x", label =r"$E = %.2f\,$EeV" %Ebins[2])
    #plt.xlabel("Zenith [Deg.]")
    #plt.ylabel("$E_{rad,tot}/sin(alpha)**2 \, $[MeV]")
    ##plt.xscale("log")
    #plt.yscale("log")
    #plt.legend(ncol=1, framealpha=0.8)
    #plt.savefig(OutputPath + "_" + title + "_vs_Energy_zen%.1f_z%d.pdf" %(SelZen, SelDepth), bbox_inches = "tight")
    #plt.show()

    return np.array(Eradair_allsims[sel][:,6][arg]), np.array(Ex_depth), np.array(Ey_depth), np.array(Ez_depth), rhoXmaxall, Eradtotall

zen_sorted, Ex0, Ey0, Ez0, rhoXmaxall, Eradtotall   = PlotEradAirThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath, rho_xmax_16_5, rho_xmax_17, rho_xmax_17_5)


def fit_density_scaling(rho, Erad):

    def model(x, a):
        return a/x**2
    popt, pcov = curve_fit(model, rho, Erad)
    return popt
arg = np.argsort(rhoXmaxall)    
rhoXmaxall = rhoXmaxall[arg]
Eradtotall = Eradtotall[arg]
y0 = fit_density_scaling(rhoXmaxall, Eradtotall)

print(y0, 2)
density_fit = y0/rhoXmaxall**2
plt.scatter(rhoXmaxall, Eradtotall, marker = "+", label = "Data", s =70, color="black")
scale_factor = 1.03
slope = y0/scale_factor
plt.plot(rhoXmaxall, density_fit/1.03, label = r"Fit: $E_{rad} =%.2f \, \rho^{-2}$" %slope, color="#D94A38")
plt.xlabel("$\\rho_{xmax} \, $[kg/m$^3$]")
plt.ylabel(r"$E_{\rm rad}/[\mathcal{E}_p/1\, {\rm EeV} \times sin{(\alpha)}]^2 \, $[MeV]")
plt.axvspan(0.8, 0.9, color='gray', alpha=0.2)
plt.legend()
plt.savefig(OutputPath + "_" + title + "_DensityScaling_Energy_zen%.1f_z%d.pdf" %(SelZen, 100), bbox_inches = "tight")
plt.show()

np.savetxt("./EradAirThetaScaling_rhoXmax.txt", np.array([rhoXmaxall, Eradtotall]), header = "rhoXmaxall[kg/m3], Eradtotall[MeV]")