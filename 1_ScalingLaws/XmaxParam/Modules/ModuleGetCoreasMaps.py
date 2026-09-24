import numpy  as np
import matplotlib.pyplot as plt
import h5py


def GetCoreasTracesfromHDF5(HDF5filepath):
    k =0
    Traces_C = dict()
    with h5py.File(HDF5filepath, "r") as f:
        #print("keys")
        observers_coreas = f["CoREAS"]["observers"]  # Navigate to the group
        keys_coreas = list(observers_coreas.keys())  # List all datasets inside
        #print(keys_coreas)
        #print("keys")
        #print("Available keys:", keys)
        for key_c in keys_coreas:
            #print(key_c, key_g)
            Traces_C[k] = observers_coreas[key_c][()]
            k = k +1
        
        return Traces_C
    

def GetPeakTraces(Traces):

    Nant = len(Traces)
    #print(Nant)
    Etot = np.zeros(Nant)
    Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    
    for i in range(Nant):
        
        Ex[i] = max(abs(Traces[i][:,1]))
        Ey[i] = max(abs(Traces[i][:,2]))
        Ez[i] = max(abs(Traces[i][:,3]))
        Etot[i] = max(np.sqrt((Traces[i][:,1])**2 + \
            (Traces[i][:,2])**2 + (Traces[i][:,3])**2)) 
        
    return Ex, Ey, Ez, Etot

def PlotCoreasMaps(Shower, Etot):

    Pos = Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths() 
    for i in range(Nlay):
        sel = (Pos[:,2] == Depths[i])
        Elog = np.log10(Etot[sel] +1)

        plt.scatter(Pos[sel,0], Pos[sel,1], \
                    c= Etot[sel]+1, cmap = "jet", s=2, edgecolors='k', linewidth=0.2)
        cbar = plt.colorbar()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        cbar.set_label("$E$ [$\mu V/m$]")
        plt.legend(["Depth = %.f m" %(Depths[i])], loc ="upper right")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.title("Full band Efield (E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(Shower.energy, Shower.zenith), size =14)
        OutputFolder = "/Users/chiche/Desktop/AntennaGridMaps/"
        #plt.savefig(OutputFolder + "Efield_E%.2f_th%.1f_|z|%.d_smalldots.pdf"  %(Shower.energy, Shower.zenith, Depths[i]), bbox_inches = "tight")
        plt.show()

    return
