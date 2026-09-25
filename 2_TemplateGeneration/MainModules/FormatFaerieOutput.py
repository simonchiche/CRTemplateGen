import numpy as np

def Traces_cgs_to_si(Traces):
    
    Nant = len(Traces)
    k = 29979.24588*1e6 # from statVolt/cm to µV/m 
    
    for i in range(Nant):
        #print(Traces[i].shape, i)
        Traces[i][:,1:] = Traces[i][:,1:]*k
    
    return Traces