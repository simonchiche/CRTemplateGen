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
SimDir =   "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "Erad_filtered"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims = []
Eradice_allsims = []
Eradtot = []
XmaxAll =[]
counter = 0
Etotairint_all, Etoticeint_all = dict(), dict()
for simpath in SimpathAll:

    print(simpath.split("/")[-1])
    
    Shower = CreateShowerfromHDF5(simpath)
    XmaxAll.append(Shower.xmax)
    # =============================================================================
    #                              Load Traces
    # =============================================================================

    energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
   
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
    ExC_int, EyC_int, EzC_int, EtotC_int, peakTime = Shower.GetIntTraces(Traces_C)
    ExG_int, EyG_int, EzG_int, EtotG_int, peakTime = Shower.GetIntTraces(Traces_G)
    sel = (Pos[:,2] == 3116)
    Etotairint_all[simpath] = EtotC_int[sel]
    Etoticeint_all[simpath] = EtotG_int[sel]

    #sys.exit(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradair_allsims.append(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradice_allsims.append(Shower.GetRadiationEnergyGeneric(Traces_G))
    Eradtot.append(Shower.GetRadiationEnergyGeneric(Traces_C))
 
Eradair_allsims = np.concatenate(Eradair_allsims, axis =0)
Eradice_allsims = np.concatenate(Eradice_allsims, axis =0)

# =============================================================================
#                             Plots
# =============================================================================


if(SimDir == "DeepCRFluctuations"):
    Depths = np.unique(Eradair_allsims[:,4])
    SelDepth = Depths[0]
    print(SelDepth)
    std_Eradair_x, std_Eradair_y, std_Eradair_z, std_Eradair_tot = np.std(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,0]), \
    np.std(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,1]), np.std(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,2]), \
    np.std(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,3])

    std_Eradice_x, std_Eradice_y, std_Eradice_z, std_Eradice_tot = np.std(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,0]), \
    np.std(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,1]), np.std(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,2]), \
    np.std(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,3])

    mean_Eradice_x, mean_Eradice_y, mean_Eradice_z, mean_Eradice_tot = np.mean(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,0]), \
    np.mean(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,1]), np.mean(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,2]), \
    np.mean(Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,3])

    mean_Eradair_x, mean_Eradair_y, mean_Eradair_z, mean_Eradair_tot = np.mean(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,0]), \
    np.mean(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,1]), np.mean(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,2]), \
    np.mean(Eradair_allsims[Eradair_allsims[:,4] == SelDepth][:,3])

    std_Eradair_x, std_Eradair_y, std_Eradair_z, std_Eradair_tot = \
        std_Eradair_x/mean_Eradair_x, std_Eradair_y/mean_Eradair_y, std_Eradair_z/mean_Eradair_z, std_Eradair_tot/mean_Eradair_tot
    
    std_Eradair_x, std_Eradair_y, std_Eradair_z, std_Eradair_tot = std_Eradair_x/np.sqrt(2), std_Eradair_y/np.sqrt(2), std_Eradair_z/np.sqrt(2), std_Eradair_tot/np.sqrt(2)
    
    std_Eradice_x, std_Eradice_y, std_Eradice_z, std_Eradice_tot = \
        std_Eradice_x/mean_Eradice_x, std_Eradice_y/mean_Eradice_y, std_Eradice_z/mean_Eradice_z, std_Eradice_tot/mean_Eradice_tot
    
    std_Eradice_x, std_Eradice_y, std_Eradice_z, std_Eradice_tot = std_Eradice_x/np.sqrt(2), std_Eradice_y/np.sqrt(2), std_Eradice_z/np.sqrt(2), std_Eradice_tot/np.sqrt(2)

    #plt.hist(Etotairint_all[SimpathAll[0]], bins=20, alpha=0.5, label="In-air")
    #plt.hist(Etotairint_all[SimpathAll[1]], bins=20, alpha=0.5, label="In-air")
    #plt.hist(Etotairint_all[SimpathAll[2]], bins=20, alpha=0.5, label="In-air")
    #plt.hist(Etotairint_all[SimpathAll[3]], bins=20, alpha=0.5, label="In-air")
    #plt.hist(Etotairint_all[SimpathAll[4]], bins=20, alpha=0.5, label="In-air")
    #print(SimpathAll)

    #print(np.median(Etotairint_all[SimpathAll[0]]), np.median(Etotairint_all[SimpathAll[1]]), np.median(Etotairint_all[SimpathAll[2]]), np.median(Etotairint_all[SimpathAll[3]]), np.median(Etotairint_all[SimpathAll[4]]))
    #MedianAirAll = np.array([np.median(Etotairint_all[SimpathAll[0]]), np.median(Etotairint_all[SimpathAll[2]]), np.median(Etotairint_all[SimpathAll[3]]), np.median(Etotairint_all[SimpathAll[4]])])
    #MedianIceAll = np.array([np.median(Etoticeint_all[SimpathAll[0]]), np.median(Etoticeint_all[SimpathAll[2]]), np.median(Etoticeint_all[SimpathAll[3]]), np.median(Etoticeint_all[SimpathAll[4]])]) 


    #print(XmaxAll)
    #print(MedianIceAll)

    #print(2*np.std(MedianAirAll)/np.mean(MedianAirAll)/np.sqrt(2))
    #print(2*np.std(MedianIceAll)/np.mean(MedianIceAll)/np.sqrt(2))

    #plt.hist(Etoticeint_all[SimpathAll[0]], bins=20, alpha=0.5, label="In-ice")
    #plt.hist(Etoticeint_all[SimpathAll[1]], bins=20, alpha=0.5, label="In-ice")
    #plt.hist(Etoticeint_all[SimpathAll[2]], bins=20, alpha=0.5, label="In-ice")
    #plt.hist(Etoticeint_all[SimpathAll[3]], bins=20, alpha=0.5, label="In-ice")
    #plt.hist(Etoticeint_all[SimpathAll[4]], bins=20, alpha=0.5, label="In-ice")

    #print(np.median(Etoticeint_all[SimpathAll[0]]), np.median(Etoticeint_all[SimpathAll[1]]), np.median(Etoticeint_all[SimpathAll[2]]), np.median(Etoticeint_all[SimpathAll[3]]), np.median(Etoticeint_all[SimpathAll[4]]))

    


    std_Eradair = np.array([std_Eradair_x, std_Eradair_y, std_Eradair_z, std_Eradair_tot])
    std_Eradice = np.array([std_Eradice_x, std_Eradice_y, std_Eradice_z, std_Eradice_tot])

    print(Shower.xmax)
    Etotair= Eradair_allsims[:,3]*1000
    plt.hist(Etotair, bins=20, alpha=0.5, label="In-air")
    print(np.std(Etotair))
    print(np.mean(Etotair))
    print(np.std(Etotair)/np.mean(Etotair)/np.sqrt(2))

    np.savetxt("./Data/Eradair_fluctuations.txt", std_Eradair)
    np.savetxt("./Data/Eradice_fluctuations.txt", std_Eradice)

#Eradair_fluctuations = np.loadtxt("./Data/" + "Eradair_fluctuations.txt")
#Eradice_fluctuations = np.loadtxt("./Data/" + "Eradice_fluctuations.txt")



SelDepth = Depths[0]
Etotair=Eradice_allsims[Eradice_allsims[:,4] == SelDepth][:,3]*1000
print(Etotair)
print(np.sort(Etotair)[:3])
Etotair = np.sort(Etotair)[:]  
plt.hist(Etotair, bins=20, alpha=0.5, label="In-air")
print(np.std(Etotair))
print(np.mean(Etotair))
print(np.std(Etotair)/np.mean(Etotair)/np.sqrt(2))


# In-air radiation energy vs depth
SelZen = 0
title = "In-air"
PlotEradDepthScaling(Eradair_allsims, SelZen, title, OutputPath)

# In-ice radiation energy vs depth
SelZen = 0
title = "In-ice"
PlotEradDepthScaling(Eradice_allsims, SelZen, title, OutputPath)

# In-air radiation energy vs zenith angle
SelE = 0.0316
title = "In-air"
PlotEradThetaScaling(Eradair_allsims, Depths, SelE, SelZen, title, OutputPath)

# In-ice radiation energy vs zenith angle
SelE = 0.316
title = "In-ice"
PlotEradThetaScaling(Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)

# Air, Ice radiation energy vs zenith angle

def PlotEradAirIceThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        arg = np.argsort(Eradair_allsims[sel][:,6])

        plt.errorbar(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,0][arg], yerr=0.15*Eradair_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{x,\mathrm{air}}$", color="#0072B2", linestyle='dashed')
        plt.errorbar(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,1][arg], yerr=0.15*Eradair_allsims[sel][:,1][arg], label ="$E^{\mathrm{rad}}_{y,\mathrm{air}}$", color="#E69F00", linestyle='dashed')
        plt.errorbar(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,2][arg], yerr=0.15*Eradair_allsims[sel][:,2][arg], label ="$E^{\mathrm{rad}}_{z,\mathrm{air}}$", color="#CC79A7", linestyle='dashed')
        plt.errorbar(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,0][arg], yerr=0.3*Eradice_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{x,\mathrm{ice}}$", color="#0072B2")
        plt.errorbar(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,1][arg], yerr=0.3*Eradice_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{y,\mathrm{ice}}$", color="#E69F00")
        plt.errorbar(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,2][arg], yerr=0.3*Eradice_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{z,\mathrm{ice}}$", color="#CC79A7")
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{\mathrm{rad}} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend(ncol=2, framealpha=0.8)
        plt.grid(alpha=0.3)
        plt.title(" $E=10^{17.5}\,$eV, Depth =%d m" %(3216-Depths[i]), fontsize=14)
        print(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf")
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return
### Article Plot
title ="AirIce"
PlotEradAirIceThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)


# In-air radiation energy vs primary energy
SelDepth = 3116
title = "In-air"
PlotEradEnergyScaling(Eradair_allsims, SelDepth, title, Shower, OutputPath)

def GetMeanEradScalingVsE(Eradair_allsims, Eradice_allsims, SelDepth, title, OutputPath):
    ZenithAll = np.unique(Eradair_allsims[:,6])

    Ys =[]
    for i in range(1,8,1):#len(ZenithAll)):
        sel = (Eradair_allsims[:,6] == ZenithAll[i]) & (Eradair_allsims[:,4] == SelDepth)
        arg = np.argsort(Eradair_allsims[sel][:,5])
        X = Eradair_allsims[sel][:,5][arg]
        y = np.sqrt(Eradair_allsims[sel][:,3][arg]/min(Eradair_allsims[sel][:,3][arg]))
        Ys.append(y)

    ZenithAll = np.unique(Eradice_allsims[:,6])

    Ysice =[]
    for i in range(1,8,1):#len(ZenithAll)):
        sel = (Eradice_allsims[:,6] == ZenithAll[i]) & (Eradice_allsims[:,4] == SelDepth)
        arg = np.argsort(Eradice_allsims[sel][:,5])
        X = Eradice_allsims[sel][:,5][arg]
        y = np.sqrt(Eradice_allsims[sel][:,3][arg]/min(Eradice_allsims[sel][:,3][arg]))
        Ysice.append(y)

    Ys = np.vstack(Ys)         # shape (n_curves, Npoints)
    y_mean = Ys.mean(axis=0)
    y_std = Ys.std(axis=0)


    Ysice = np.vstack(Ysice)         # shape (n_curves, Npoints)
    y_mean_ice = Ysice.mean(axis=0)
    y_std_ice = Ysice.std(axis=0)

    return X, y_mean, y_std, y_mean_ice, y_std_ice/1.5

# Mean Erad scaling vs E
SelDepth = 3116
X, y_mean, y_std, y_mean_ice, y_std_ice = \
    GetMeanEradScalingVsE(Eradair_allsims, Eradice_allsims, SelDepth, title, OutputPath)

PlotMeanEradScalingVsE(X, y_mean, y_std, y_mean_ice, y_std_ice, SelDepth, Shower, OutputPath)

# In-ice radiation energy vs primary energy
SelDepth = 3116
title = "In-ice"
PlotEradEnergyScaling(Eradice_allsims, SelDepth, title, Shower, OutputPath)


# In-air Energy scaling as a function of depth
title = "In-air"
Eindex = 2 #E17.5/E17
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)

title = "In-air"
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)

title = "In-ice"
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)


# In-ice Energy scaling as a function of depth
title = "In-ice"
Eindex = 2 #E17.5/E17
PlotEradIceEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)
Eindex = 1 #E17/E16.5
PlotEradIceEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)

SelE = 0.316
title ="Air_vs_Ice_vs_Theta"
PlotEradtotThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)




################ Test Ground Particle Energy Scaling vs Depth #####################


# Ground Particle Energy vs Zenith Angle at fixed Primary Energy
DataAll = \
    glob.glob("/Users/chiche/Desktop/DeepCrAnalysis/8_SideStudies/ShowerPhysics/Data/GroundParticleFiles/*")
EGroundPart, EprimaryAllpart, ZenithAllpart = GetGroundParticleEnergy(DataAll)
SelE = max(EprimaryAllpart)


def GetEgroundPart_E(EGroundPart, EprimaryAllpart, ZenithAllpart, SelE):
    Ebins = np.unique(EprimaryAllpart)

    ZenE = ZenithAllpart[EprimaryAllpart==SelE]
    EGroundPartE = EGroundPart[EprimaryAllpart==SelE]

    argzensort = np.argsort(ZenE)
    ZenE = ZenE[argzensort]
    EGroundPartE = EGroundPartE[argzensort]

    return ZenE, EGroundPartE


ZenE, EGroundPartE = GetEgroundPart_E(EGroundPart, EprimaryAllpart, ZenithAllpart, SelE)
PlotGroundParticleEVsZenith(ZenE, EGroundPartE, SelE)

# In-ice radiation  Energy vs Zenith Angle at fixed Primary Energy
#SelE = max(EprimaryAllpart)

def EradicevsZenE(Eradice_allsims, Depths, SelE):
    
    Ebins = np.unique(Eradice_allsims[:,5])

    sel = (Eradice_allsims[:,4] == min(Depths))  & (Eradice_allsims[:,5] ==SelE)
    arg = np.argsort(Eradice_allsims[sel][:,6])

    ZenE_Erad = Eradice_allsims[sel][:,6][arg]
    EradiceE = Eradice_allsims[sel][:,3][arg]

    return ZenE_Erad, EradiceE


ZenE_Erad, EradiceE = EradicevsZenE(Eradice_allsims, Depths, SelE)
PlotEradIcevsZenE(ZenE_Erad, EradiceE)

# In-ice radiation Energy vs Ground Particle Energy at fixed Primary Energy
EGroundPartE, EfieldIceE, X, Ylinear = GetEradvsEgroundPart(EGroundPartE, EradiceE)

def PlotEradIcevsEgroundPart(EGroundPartE, EfieldIceE, OutputPath):
    from scipy.optimize import curve_fit
    def ModelFunc(x, a,b):
        return  a*x**b
    popt, pcov = curve_fit(ModelFunc, EGroundPartE, EfieldIceE)

    plt.scatter(EGroundPartE, EfieldIceE, marker='*', label="$E_{\mathrm{rad}}^{\mathrm{ice}}$", s=50)
    #plt.plot(X, Ylinear+0.1e-7, 'r--', label='Linear scaling')
    plt.plot(EGroundPartE, ModelFunc(EGroundPartE, *popt), label= r"$a (E_{\mathrm{ground}}^{\mathrm{part}})^{b}$", color="red")
    plt.xlabel('$E_{\mathrm{part}}^{\mathrm{ground}}\,[\mathrm{GeV}]$')
    plt.ylabel('$\sqrt{E_{\mathrm{rad}}^{\mathrm{ice}}\,[MeV]}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(OutputPath + "EfieldIce_vs_EgroundPart.pdf", bbox_inches = 'tight')
    plt.show()

    a_err, b_err = np.sqrt(np.diag(pcov))
    yfit = ModelFunc(EGroundPartE, *popt)
    ss_res = np.sum((EfieldIceE - yfit)**2)
    ss_tot = np.sum((EfieldIceE - np.mean(EfieldIceE))**2)
    r2 = 1 - ss_res/ss_tot


    print("Fit parameters: a = %.3e, b = %.3f" % (popt[0], popt[1]))
    print(a_err, b_err, r2)
    return  a_err, b_err, r2

a_err, b_err, r2 = PlotEradIcevsEgroundPart(EGroundPartE, EfieldIceE, OutputPath)





## Fixer les unités
# obtenir les plots sur le zenith
# Air/ice ratio