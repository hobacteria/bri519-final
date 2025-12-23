"""
Data Loading Module
Original code from midterm notebook - Cells 7, 10, 11, 12, 13
"""

import numpy as np
import scipy.io


def load_imports():
    """
    Load all necessary imports
    Original code from notebook Cell 7
    """
    import numpy as np
    import scipy.io
    import scipy.signal
    import scipy.stats
    import matplotlib.pyplot as plt
    from collections import defaultdict
    return np, scipy.io, scipy.signal, scipy.stats, plt, defaultdict


def load_data(mat_file_path='mouseLFP.mat'):
    """
    Load the dataset and assign key data and parameters
    Original code from notebook Cell 10
    
    Args:
        mat_file_path: Path to the mouseLFP.mat file
        
    Returns:
        DATA: Loaded data array
        dataSamples: Number of data samples
    """
    # Original code from Cell 10
    mat_data = scipy.io.loadmat(mat_file_path)
    DATA = mat_data['DATA']
    dataSamples = DATA[0,0].shape[1]
    
    return DATA, dataSamples


def inspect_data(DATA):
    """
    Inspect data structure
    Original code from notebook Cells 11, 12, 13
    """
    # Original code from Cell 11
    print(DATA.shape)
    print(DATA[0][6]) ## last data is name of trial
    print(DATA[0][0].shape)
    
    # Original code from Cell 12
    for i in range(6):
        print(DATA[0,i].shape)
    
    # Original code from Cell 13
    for i in range(4):
        print(np.unique(DATA[i,4])) ## tone

