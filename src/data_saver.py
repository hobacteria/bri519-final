"""
Data Saving Module
Original code from midterm notebook - Cell 52
"""

import pickle
import numpy as np


def save_results(rawData, filteredData, toneIndices, outlierIndices, 
                 trialsBeforeRejection, trialsAfterRejection, 
                 cutoffFrequency, fs, stimOnset, stimOffset, numSessions, dataSamples,
                 output_dir='results'):
    """
    Save the results including raw data, low-pass filtered data, and main analysis results
    Original code from notebook Cell 52
    
    Args:
        rawData: Dictionary with cleaned raw data
        filteredData: List of dictionaries with filtered data
        toneIndices: Dictionary with tone indices
        outlierIndices: Dictionary with outlier indices
        trialsBeforeRejection: Dictionary with trial counts before rejection
        trialsAfterRejection: Dictionary with trial counts after rejection
        cutoffFrequency: Cutoff frequency
        fs: Sampling frequency
        stimOnset: Stimulus onset time (ms)
        stimOffset: Stimulus offset time (ms)
        numSessions: Number of sessions
        dataSamples: Number of data samples
        output_dir: Output directory for saving results
    """
    # Original code from Cell 52
    import pickle

    results = {
        'rawData': rawData,
        'filteredData': filteredData,
        'toneIndices': toneIndices,
        'outlierIndices': outlierIndices,
        'trialsBeforeRejection': trialsBeforeRejection,
        'trialsAfterRejection': trialsAfterRejection,
        'parameters': {
            'cutoffFrequency': cutoffFrequency,
            'fs': fs,
            'stimOnset': stimOnset,
            'stimOffset': stimOffset,
            'numSessions': numSessions,
            'dataSamples': dataSamples
        }
    }

    with open(f'{output_dir}/analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    np.savez(f'{output_dir}/analysis_results.npz',
             rawData=rawData,
             filteredData=filteredData,
             toneIndices=toneIndices)

