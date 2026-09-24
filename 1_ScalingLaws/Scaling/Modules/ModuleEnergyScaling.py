import numpy  as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit


### TEST0: Erad/Ep^2 vs E
def model_linear(x, k):
    return k * x

def PlotEradEnergyScaling_zenbin(Erad_allsims, SelDepth, title, Shower, OutputPath):

    ZenithAll = np.unique(Erad_allsims[:,6])

    for i in range(1,8,1):#len(ZenithAll)):
        sel = (Erad_allsims[:,6] == ZenithAll[i]) & (Erad_allsims[:,4] == SelDepth)

        arg = np.argsort(Erad_allsims[sel][:,5])
        EnergyBins= np.unique(Erad_allsims[sel][:,5])
        Erad_lin = np.sqrt(Erad_allsims[sel][:,3][arg])
        plt.errorbar(EnergyBins, Erad_lin, yerr=0.1*Erad_lin, fmt="-o",\
                 label ="$\\theta =%.d^{\circ}$" %ZenithAll[i])
        
        popt, pcov = curve_fit(model_linear, EnergyBins, Erad_lin)

        k=popt[0]
        plt.plot(EnergyBins, k*EnergyBins)
        plt.xlabel("Energy [EeV]")
        plt.ylabel("$E_{rad} \,$[eV]")
        plt.legend()
        plt.yscale('log')
        plt.xscale('log') 
        plt.title(title + ", Depth$=%.d\,$m" %(Shower.glevel- SelDepth), fontsize=14)
        #plt.savefig(OutputPath + "_" + title + "_vs_E_|z|%.d.pdf" %SelDepth, bbox_inches = "tight")
        plt.show()
    return

def PlotEradEnergyScaling_normalized(Erad_allsims, SelDepth, title, Shower, OutputPath):

    ZenithAll = np.unique(Erad_allsims[:,6])

    for i in range(1,8,1):#len(ZenithAll)):
        sel = (Erad_allsims[:,6] == ZenithAll[i]) & (Erad_allsims[:,4] == SelDepth)

        arg = np.argsort(Erad_allsims[sel][:,5])
        EnergyBins= np.unique(Erad_allsims[sel][:,5])
        Erad_lin = np.sqrt(Erad_allsims[sel][:,3][arg])
        Erad_lin = Erad_allsims[sel][:,3][arg]

        popt, pcov = curve_fit(model_linear, EnergyBins, Erad_lin)

        k=popt[0]
        plt.errorbar(EnergyBins**2, Erad_lin/Erad_lin[1], yerr=0.1*Erad_lin/(Erad_lin[1]), fmt="-o",\
                 label ="$\\theta =%.d^{\circ}$" %ZenithAll[i])
        

    #plt.plot(EnergyBins, k*EnergyBins/(k*EnergyBins[0]))
    plt.xlabel("(Energy$)^{2}$ [EeV$^{2}$]")
    plt.ylabel("$E_{rad} \,$[eV]")
    plt.legend()
    plt.yscale('log')
    plt.xscale('log') 
    plt.title(title + ", Depth$=%.d\,$m" %(Shower.glevel- SelDepth), fontsize=14)
    #plt.savefig(OutputPath + "_" + title + "_vs_E_|z|%.d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()
    return

def PlotEradEnergyScaling_vs_theta(Erad_allsims, SelDepth, title, Shower, OutputPath):

    ZenithAll = np.unique(Erad_allsims[:,6])
    EnergyBins= np.unique(Erad_allsims[:,5])

    for i in range(len(EnergyBins)):
        sel = (Erad_allsims[:,5] == EnergyBins[i]) & (Erad_allsims[:,4] == SelDepth)

        arg = np.argsort(Erad_allsims[sel][:,6])

        Erad_lin = np.sqrt(Erad_allsims[sel][:,3][arg])

        plt.errorbar(ZenithAll[1:], Erad_lin[1:]/(EnergyBins[i]), yerr=0.1*Erad_lin[1:]/(0.9*EnergyBins[i]), fmt="-o",\
                 label ="$E =%.2f $ EeV" %EnergyBins[i])
        

    #plt.plot(EnergyBins, k*EnergyBins/(k*EnergyBins[0]))
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel("$E_{rad} \,$[eV]")
    plt.legend()
    #plt.yscale('log')
    #plt.xscale('log') 
    plt.title(title + ", Depth$=%.d\,$m" %(Shower.glevel- SelDepth), fontsize=14)
    #plt.savefig(OutputPath + "_" + title + "_vs_E_|z|%.d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()
    return
