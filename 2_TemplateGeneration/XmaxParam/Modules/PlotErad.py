import numpy as np
import matplotlib.pyplot as plt

def PlotEradThetaScaling(Erad_allsims, Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    for i in range(len(Depths)):

        sel = (Erad_allsims[:,4] == Depths[i]) & (Erad_allsims[:,5] == SelE)
        
        arg = np.argsort(Erad_allsims[sel][:,6])
        plt.plot(Erad_allsims[sel][:,6][arg], Erad_allsims[sel][:,0][arg], label ="$E^{rad}_x$")
        plt.plot(Erad_allsims[sel][:,6][arg], Erad_allsims[sel][:,1][arg], label ="$E^{rad}_y$")
        plt.plot(Erad_allsims[sel][:,6][arg], Erad_allsims[sel][:,2][arg], label ="$E^{rad}_z$")
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{rad} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend()
        plt.title(title + " $E=%.2f\,$ EeV |z| =%d m" %(SelE, Depths[i]))
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return

def PlotEradDepthScaling(Erad_allsims, SelZen, title, OutputPath):

    EnergyAll = np.unique(Erad_allsims[:,5])
    for i in range(len(EnergyAll)):
        SelE = EnergyAll[i]
        sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
        plt.scatter(Erad_allsims[sel][:,4], Erad_allsims[sel][:,3],\
                     label = "E = %.2f EeV" %EnergyAll[i])
    plt.yscale("log")
    #plt.ylim(min(Erad_allsims[:,5])/5, max(Erad_allsims[:,5])*5)
    plt.ylabel("$E_{rad} \, $[MeV]")
    plt.title(title + " $\\theta = %.d^{\circ}$" %(SelZen))
    plt.xlabel("Depth [m]")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9), framealpha=0.5)
    plt.savefig(OutputPath + "_" + title + "_vs_Depth_th%.d.pdf" %SelZen, bbox_inches = "tight")
    plt.show()
    return


def PlotEradEnergyScaling(Erad_allsims, SelDepth, title, Shower, OutputPath):

    ZenithAll = np.unique(Erad_allsims[:,6])
    x_lin = np.array([0.0316,0.316])
    y_lin = np.array([1,10])
    for i in range(1,8,1):#len(ZenithAll)):
        sel = (Erad_allsims[:,6] == ZenithAll[i]) & (Erad_allsims[:,4] == SelDepth)

        arg = np.argsort(Erad_allsims[sel][:,5])
        plt.plot(Erad_allsims[sel][:,5][arg], np.sqrt(Erad_allsims[sel][:,3][arg]/min(Erad_allsims[sel][:,3][arg])),\
                 label ="$\\theta =%.d^{\circ}$" %ZenithAll[i])
        plt.plot(x_lin, y_lin, '--', color ='red', linewidth=2)
    plt.xlabel("Energy [EeV]")
    plt.ylabel("$E_{rad} \,$[eV]")
    plt.legend()
    #plt.yscale('log')
    plt.title(title + ", Depth$=%.d\,$m" %(Shower.glevel- SelDepth), fontsize=14)
    #plt.savefig(OutputPath + "_" + title + "_vs_E_|z|%.d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()
    return


def PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath):

    ZenithAll =  np.unique(Eradair_allsims[:,6])
    EnergyAll =  np.unique(Eradair_allsims[:,5])
    
    for i in range(len(ZenithAll)):

        selE1 = (Eradair_allsims[:,6] == ZenithAll[i]) &  (Eradair_allsims[:,5] == EnergyAll[Eindex])
        EindexLow = (Eindex -1)
        selE2 = (Eradair_allsims[:,6] == ZenithAll[i]) &  (Eradair_allsims[:,5] == EnergyAll[EindexLow])
        EscalingRatio = Eradair_allsims[:,3][selE1]/ Eradair_allsims[:,3][selE2]

        plt.plot(3216 -Eradair_allsims[:,4][selE1], EscalingRatio, label = "$\\theta =%.d^{\circ}$" %ZenithAll[i], marker='o', linestyle='-')
    plt.legend(loc='upper left')
    plt.xlabel("Depth [m]")
    plt.ylabel("$E%.2f/E%.2f$" %(EnergyAll[Eindex], EnergyAll[EindexLow]))
    if(EnergyAll[Eindex]==0.316):
        plt.ylabel("$E_{\mathrm{rad}}(E=10^{17.5}\,\mathrm{eV})/E_{\mathrm{rad}}(E=10^{17}\,\mathrm{eV})$")
    plt.title(title, fontsize=14)
    plt.savefig(OutputPath + "_" + title + "Escaling%.2f_vs_Depth.pdf" %EnergyAll[Eindex], bbox_inches = "tight")
    plt.show()

    return



def PlotAirIceEradRatiovsTheta(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, OutputPath):
    sel = (Eradair_allsims[:,6] == SelZen) & (Eradair_allsims[:,5] == SelE)
    
    
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        EradAirIceRatio_x = Eradair_allsims[sel][:,0]/ Eradice_allsims[sel][:,0]
        EradAirIceRatio_y = Eradair_allsims[sel][:,1]/ Eradice_allsims[sel][:,1]
        EradAirIceRatio_z = Eradair_allsims[sel][:,2]/ Eradice_allsims[sel][:,2]

        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_x[arg], label ="$E^{rad, air}_x/E^{rad, ice}_x$")
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_y[arg], label ="$E^{rad, air}_y/E^{rad, ice}_y$")
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_z[arg], label ="$E^{rad, air}_z/E^{rad, ice}_z$")
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{rad}^{air}/E_{rad}^{ice}\,$[50-1000 MHz]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend()
        plt.title("$E=%.2f\,$ EeV |z| =%d m" %(SelE, Depths[i]))
        plt.savefig(OutputPath + "air_ice_ratio_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return


def PlotAirIceEradRatiovsThetavsE(Eradair_allsims, Eradice_allsims, SelDepth, OutputPath):

    EnergyAll = np.unique(Eradair_allsims[:,5])    
    
    for i in range(len(EnergyAll)):
        sel = (Eradair_allsims[:,4] == SelDepth) & (Eradair_allsims[:,5] == EnergyAll[i])
        
        #EradAirIceRatio_x = Eradair_allsims[sel][:,0]/ Eradice_allsims[sel][:,0]
        #EradAirIceRatio_y = Eradair_allsims[sel][:,1]/ Eradice_allsims[sel][:,1]
        #EradAirIceRatio_z = Eradair_allsims[sel][:,2]/ Eradice_allsims[sel][:,2]
        EradAirIceRatio_tot = Eradair_allsims[sel][:,3]/ Eradice_allsims[sel][:,3]

        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_tot[arg], label ="E= $%.2f$ EeV" %EnergyAll[i])
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_x[arg], label ="$E^{rad, air}_x/E^{rad, ice}_x$")
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_y[arg], label ="$E^{rad, air}_y/E^{rad, ice}_y$")
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_z[arg], label ="$E^{rad, air}_z/E^{rad, ice}_z$")
    #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
    plt.yscale("log")
    #plt.ylim(min(data)/5, max(data)*5)
    plt.ylabel("$E_{rad}^{air}/E_{rad}^{ice}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.legend()
    plt.title("$|z| =%d$ m" %(SelDepth))
    plt.savefig(OutputPath + "air_ice_ratio_vs_theta_vsE_z%d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()

    return

def PlotHpoleVpoleEradRatiovsThetavsE(Erad_allsims, SelDepth, title, OutputPath):

    EnergyAll = np.unique(Erad_allsims[:,5])    
    
    for i in range(len(EnergyAll)):
        sel = (Erad_allsims[:,4] == SelDepth) & (Erad_allsims[:,5] == EnergyAll[i])
        
        EradHpole_tot = (Erad_allsims[sel][:,0] + Erad_allsims[sel][:,1])/2.0
        EradVpole_tot = Erad_allsims[sel][:,2]
        EradHpoleVpoleRatio_tot = EradHpole_tot/EradVpole_tot

        arg = np.argsort(Erad_allsims[sel][:,6])
        plt.plot(Erad_allsims[sel][:,6][arg], EradHpoleVpoleRatio_tot[arg], label ="$E= %.2f$" %EnergyAll[i])
    plt.yscale("log")
    #plt.ylim(min(data)/5, max(data)*5)
    plt.ylabel("$E_{rad}^{hpole}/E_{rad}^{vpole}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.grid()
    plt.legend()
    plt.title(title + " |z| =%d m" %(SelDepth))
    plt.savefig(OutputPath + "_" + title + "_Hpole_over_Vpole_vs_E_vs_zenith_z%d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()

    return

def PlotEradtotThetaScaling(Eradair_allsims, Eradice_allsims,  Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,3][arg], label ="In-air")
        plt.plot(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,3][arg], label ="In-ice")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{rad} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend()
        selDepth = 3216- Depths[i]
        plt.title( " $E=%.2f\,$ EeV Depth =%d m" %(SelE,selDepth), fontsize =12)
        plt.grid()
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return

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

    return X, y_mean, y_std, y_mean_ice, y_std_ice

def PlotMeanEradScalingVsE(X, y_mean, y_std, y_mean_ice, y_std_ice, SelDepth, Shower, OutputPath):
   
    x_lin = np.array([0.0316,0.316])
    y_lin = np.array([1,10])

    plt.plot(x_lin, y_lin, '--', color ='black', linewidth=4, label='linear scaling', )

    plt.errorbar(X, y_mean, yerr=y_std, fmt='-s', label='In-air', linewidth=2, color='red')
    plt.errorbar(X, y_mean_ice, yerr=y_std_ice, fmt='-s', label='In-ice', linewidth=2, color ='blue')
    plt.xlabel('Primary Energy [EeV]')
    plt.ylabel('$\sqrt{E_{rad}/E_{rad}[E=10^{16.5}\, \mathrm{eV}]}$')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    Depth = Shower.glevel - SelDepth
    plt.savefig(OutputPath + f'MeanEradScalingVsE_Depth{SelDepth}.pdf', bbox_inches='tight')
    plt.show()

    return

def PlotEradIceEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath):

    ZenithAll =  np.unique(Eradair_allsims[:,6])
    EnergyAll =  np.unique(Eradair_allsims[:,5])
    argzen = np.argsort(ZenithAll)
    ZenithAll = ZenithAll[argzen]
    
    for i in range(1, len(ZenithAll)):

        selE1 = (Eradair_allsims[:,6] == ZenithAll[i]) &  (Eradair_allsims[:,5] == EnergyAll[Eindex])
        EindexLow = (Eindex -1)
        selE2 = (Eradair_allsims[:,6] == ZenithAll[i]) &  (Eradair_allsims[:,5] == EnergyAll[EindexLow])
        EscalingRatio = Eradair_allsims[:,3][selE1]/ Eradair_allsims[:,3][selE2]
        plt.plot(3216 -Eradair_allsims[:,4][selE1][1:], EscalingRatio[1:], label = "$\\theta =%.d^{\circ}$" %ZenithAll[i], marker='o', linestyle='-')
    plt.legend(loc='upper left', framealpha=0.7)
    plt.xlabel("Depth [m]")
    plt.ylabel("$E%.2f/E%.2f$" %(EnergyAll[Eindex], EnergyAll[EindexLow]))
    if(EnergyAll[Eindex]==0.316):
        plt.ylabel("$E_{\mathrm{rad}}(E=10^{17.5}\,\mathrm{eV})/E_{\mathrm{rad}}(E=10^{17}\,\mathrm{eV})$")
    plt.title(title, fontsize=14)
    plt.savefig(OutputPath + "_" + title + "Escaling%.2f_vs_Depth.pdf" %EnergyAll[Eindex], bbox_inches = "tight")
    plt.show()

    return


def GetGroundParticleEnergy(DataAll):
    Etot = []
    ZenithAllpart =[]
    EnergyAllpart =[]
    for i in range(len(DataAll)):
        filename = DataAll[i].split("/")[-1]
        energy = float(filename.split("_")[2])
        zenith = float(filename.split("_")[3])
        EnergyAllpart.append(energy)
        ZenithAllpart.append(zenith)
        part_id, px, py, pz, x, y, t, w, E_, r_  =np.loadtxt(DataAll[i], unpack=True)
        #E_ = np.sqrt(px**2 + py**2 + pz**2)
        E_ = E_[r_<100]
        w = w[r_<100]
        w = w[E_>0.1]
        E_ = E_[E_>0.1]
        Etot.append(np.sum(E_*w))
    
    EGroundPart = np.array(Etot)
    EprimaryAllpart = np.array(EnergyAllpart)
    ZenithAllpart = np.array(ZenithAllpart)

    return EGroundPart, EprimaryAllpart, ZenithAllpart

def GetEgroundPart_E(EGroundPart, EprimaryAllpart, ZenithAllpart, SelE):
    Ebins = np.unique(EprimaryAllpart)

    ZenE = ZenithAllpart[EprimaryAllpart==SelE]
    EGroundPartE = EGroundPart[EprimaryAllpart==SelE]

    argzensort = np.argsort(ZenE)
    ZenE = ZenE[argzensort]
    EGroundPartE = EGroundPartE[argzensort]

    return ZenE, EGroundPartE

def PlotGroundParticleEVsZenith(ZenE, EGroundPartE, SelE):
# Ground Particle Energy vs Zenith Angle at fixed Primary Energy
    plt.plot(ZenE, EGroundPartE, 'o', label=f'E={SelE} EeV')
    plt.xlabel('Zenith Angle [Deg.]')
    plt.ylabel('$E_{\mathrm{part}}^{\mathrm{ground}}\,[\mathrm{GeV}]$')
    plt.show()

    return

def EradicevsZenE(Eradice_allsims, Depths, SelE):
    
    Ebins = np.unique(Eradice_allsims[:,5])

    sel = (Eradice_allsims[:,4] == min(Depths)) & (Eradice_allsims[:,5] ==SelE)
    arg = np.argsort(Eradice_allsims[sel][:,6])

    ZenE_Erad = Eradice_allsims[sel][:,6][arg]
    EradiceE = Eradice_allsims[sel][:,3][arg]

    return ZenE_Erad, EradiceE

def PlotEradIcevsZenE(ZenE_Erad, EradiceE):
    plt.plot(ZenE_Erad, EradiceE, 'o', label ="In-ice")
    plt.xlabel('Zenith Angle [Deg.]')
    plt.ylabel('$E_{\mathrm{rad}}^{\mathrm{ice}}\,[MeV]$')
    plt.legend()
    plt.show()

    return

def GetEradvsEgroundPart(EGroundPartE, EradiceE):


    argEgroundPartE_sort = np.argsort(EGroundPartE)
    EradiceE = EradiceE[argEgroundPartE_sort]
    EfieldIceE = np.sqrt(EradiceE[:-1])

    EGroundPartE = EGroundPartE[argEgroundPartE_sort][:-1]
    Ylinear = np.array([min(EfieldIceE), min(EfieldIceE)*max(EGroundPartE)/min(EGroundPartE)])
    X = np.array([min(EGroundPartE), max(EGroundPartE)])

    return EGroundPartE, EfieldIceE, X, Ylinear

def PlotEradIcevsEgroundPart(EGroundPartE, EfieldIceE, OutputPath):
    from scipy.optimize import curve_fit
    def ModelFunc(x, a):
        return  a*x
    popt, pcov = curve_fit(ModelFunc, EGroundPartE, EfieldIceE)

    plt.scatter(EGroundPartE, EfieldIceE, marker='*', label="$E_{\mathrm{rad}}^{\mathrm{ice}}$", s=50)
    #plt.plot(X, Ylinear+0.1e-7, 'r--', label='Linear scaling')
    plt.plot(EGroundPartE, ModelFunc(EGroundPartE, *popt), label= "Linear fit", color="red")
    plt.xlabel('$E_{\mathrm{part}}^{\mathrm{ground}}\,[\mathrm{GeV}]$')
    plt.ylabel('$\sqrt{E_{\mathrm{rad}}^{\mathrm{ice}}\,[MeV]}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(OutputPath + "EfieldIce_vs_EgroundPart.pdf", bbox_inches = 'tight')
    plt.show()
