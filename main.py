"""
Main execution script for LFP Analysis
This script replicates the original midterm notebook workflow
Original code structure preserved from midterm notebook
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from constants import (
    cutoffFrequency, fs, stimOnset, stimOffset, 
    numTrials, numSessions, nyquistFrequency
)
from data_loader import load_data, inspect_data
from outlier_detection import (
    find_tone_indices, visualize_raw_data, compare_tone_responses,
    calculate_high_freq_power, detect_outliers, remove_outliers,
    print_rejection_summary, visualize_cleaned_data, store_outlier_data,
    visualize_outliers
)
from filtering import setup_filtering_parameters, apply_filter, visualize_filtering
from analysis import (
    method1_psth_analysis, visualize_method1_time, visualize_method1_frequency,
    method2_correlation_analysis, visualize_method2_correlation,
    combined_sessions_analysis
)
from data_saver import save_results


def main():
    """
    Main execution function
    Replicates the original notebook workflow
    """
    print("=" * 80)
    print("LFP Data Analysis - Main Execution")
    print("=" * 80)
    
    # ===== (1) Initial Step: Constants =====
    print("\n[Step 1] Loading constants...")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Cutoff frequency: {cutoffFrequency} Hz")
    print(f"  Number of sessions: {numSessions}")
    
    # ===== (2) Loader: Load dataset =====
    print("\n[Step 2] Loading dataset...")
    mat_file_path = 'data/mouseLFP.mat'
    if not os.path.exists(mat_file_path):
        print(f"Warning: {mat_file_path} not found. Please ensure the data file is in the data/ directory.")
        return
    
    DATA, dataSamples = load_data(mat_file_path)
    print(f"  Data loaded: {DATA.shape}")
    print(f"  Data samples: {dataSamples}")
    
    # Optional: Inspect data structure
    # inspect_data(DATA)
    
    # ===== (3) Outlier Sample Rejection =====
    print("\n[Step 3] Outlier detection and removal...")
    
    # Find tone indices
    toneIndices = find_tone_indices(DATA, numSessions)
    
    # Create time array for visualization
    time_ms = np.arange(dataSamples) / fs * 1000
    
    # Visualize raw data (optional - comment out if not needed)
    # visualize_raw_data(DATA, toneIndices, numSessions)
    # compare_tone_responses(DATA, toneIndices, numSessions, time_ms, stimOnset, stimOffset)
    
    # Detect outliers
    high_freq_powers, outlier_flags = detect_outliers(DATA, toneIndices, numSessions, fs)
    
    # Remove outliers
    rawData, validToneIndices, trialsBeforeRejection, trialsAfterRejection = remove_outliers(
        DATA, toneIndices, outlier_flags, numSessions
    )
    
    # Print rejection summary
    print_rejection_summary(trialsBeforeRejection, trialsAfterRejection, numSessions)
    
    # Visualize cleaned data (optional - comment out if not needed)
    # visualize_cleaned_data(rawData, numSessions, time_ms, stimOnset, stimOffset, fs, dataSamples)
    
    # Store outlier data
    outlierIndices, outlierData = store_outlier_data(DATA, toneIndices, outlier_flags, numSessions, dataSamples)
    
    # Visualize outliers (optional - comment out if not needed)
    # visualize_outliers(outlierData, rawData, numSessions, time_ms, stimOnset, stimOffset)
    
    # ===== (4) Filtering =====
    print("\n[Step 4] Applying low-pass filter...")
    
    # Setup filtering parameters
    windLength, wind, overl, binWidth, maxFreq = setup_filtering_parameters()
    
    # Apply filter
    filteredData = apply_filter(rawData, numSessions, cutoffFrequency, nyquistFrequency)
    print("  Filtering completed")
    
    # Visualize filtering (optional - comment out if not needed)
    # visualize_filtering(rawData, filteredData, numSessions, time_ms, stimOnset, stimOffset, 
    #                    fs, windLength, wind, overl, maxFreq)
    
    # ===== (5) Main Analysis =====
    print("\n[Step 5] Main analysis...")
    
    # Method 1: PSTH-like analysis
    meanLFPs, semLFPs, responseMetrics = method1_psth_analysis(
        filteredData, numSessions, stimOnset, stimOffset, fs
    )
    
    # Visualize Method 1 (optional - comment out if not needed)
    # visualize_method1_time(meanLFPs, semLFPs, responseMetrics, numSessions, time_ms, 
    #                       stimOnset, stimOffset, fs)
    # visualize_method1_frequency(meanLFPs, numSessions, fs)
    
    # Method 2: Correlation analysis
    rSC_sessions, rSC_conditions = method2_correlation_analysis(meanLFPs, numSessions)
    
    # Visualize Method 2 (optional - comment out if not needed)
    # visualize_method2_correlation(rSC_sessions, rSC_conditions, numSessions)
    
    # Combined sessions analysis
    lfp_combined, r_combined_conditions, r_combined_freq = combined_sessions_analysis(
        filteredData, numSessions, meanLFPs, rSC_conditions, 
        stimOnset, stimOffset, fs
    )
    
    # ===== (6) Save Results =====
    print("\n[Step 6] Saving results...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    save_results(
        rawData, filteredData, toneIndices, outlierIndices,
        trialsBeforeRejection, trialsAfterRejection,
        cutoffFrequency, fs, stimOnset, stimOffset, numSessions, dataSamples,
        output_dir='results'
    )
    
    print("  Results saved to results/analysis_results.pkl and results/analysis_results.npz")
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

