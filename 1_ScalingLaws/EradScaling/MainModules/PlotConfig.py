import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime
import sys

def MatplotlibConfig(WorkPath, SimDir, BatchID):

    #region Matplotlib config (build a function)
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 14}

    plt.rc('font', **font)
    #endregion

    #region Store configuration
    # We create a directory where the outputs are stored
    date = datetime.today().strftime('%Y-%m-%d')
    WorkPath = os.getcwd().split("/")
    WorkPath = "/".join(WorkPath[5:])
    OutputPath = "/Users/chiche/Desktop/FAERIEParam/Figures/"\
          + SimDir + "/" + WorkPath +  "/" + BatchID + "/" 
    #print(OutputPath)
    #sys.exit()
    cmd = "mkdir -p " + OutputPath
    OutputPath = OutputPath + BatchID + "_"
    p =subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
    stdout, stderr = p.communicate()
    #endregion

    return OutputPath
