"""
Outlier Detection Module
Original code from midterm notebook - Cells 18, 19, 20, 23, 24, 25, 26, 27, 29
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def find_tone_indices(DATA, numSessions):
    """
    Find tone indices for each session
    Original code from notebook Cell 18
    
    Args:
        DATA: Data array
        numSessions: Number of sessions
        
    Returns:
        toneIndices: Dictionary with tone indices for each session
    """
    # Original code from Cell 18
    # Find tone indices for each session (similar to MATLAB code)
    toneIndices = {}
    for session in range(numSessions):
        tone_values = DATA[session, 4].flatten()
        unique_tones = np.unique(tone_values)
        
        # Low tone indices (minimum value)
        low_tone_idx = np.where(tone_values == np.min(unique_tones))[0]
        # High tone indices (maximum value)
        high_tone_idx = np.where(tone_values == np.max(unique_tones))[0]
        
        toneIndices[session] = {
            'low': low_tone_idx,
            'high': high_tone_idx
        }
        
        print(f"Session {session+1}:")
        print(f' unique tone : {unique_tones}')
        print(f"  Low tone trials: {len(low_tone_idx)}")
        print(f"  High tone trials: {len(high_tone_idx)}")
    
    return toneIndices


def visualize_raw_data(DATA, toneIndices, numSessions):
    """
    Visualize raw data by tone condition for each session
    Original code from notebook Cell 19
    """
    # Original code from Cell 19
    # Visualize raw data by tone condition for each session
    fig, axes = plt.subplots(numSessions, 2, figsize=(16, 12))
    fig.suptitle('Raw LFP Data by Tone Condition (All Sessions)', fontsize=16, y=0.995)

    for session in range(numSessions):
        # Low tone
        ax_low = axes[session, 0]
        low_indices = toneIndices[session]['low']
        low_data = DATA[session, 0][low_indices, :]
        
        # Plot individual trials in light gray
        for trial_idx in range(len(low_indices)):
            ax_low.plot(low_data[trial_idx, :], alpha=0.1, color='gray', linewidth=0.5)
        
        # Plot mean in black
        mean_low = np.mean(low_data, axis=0)
        ax_low.plot(mean_low, 'k', linewidth=2, label='Mean')
        ax_low.set_title(f'Session {session+1} - Low Tone (n={len(low_indices)} trials)')
        ax_low.set_xlabel('Time (samples)')
        ax_low.set_ylabel('Amplitude (mV)')
        ax_low.legend()
        ax_low.grid(True, alpha=0.3)
        
        # High tone
        ax_high = axes[session, 1]
        high_indices = toneIndices[session]['high']
        high_data = DATA[session, 0][high_indices, :]
        
        # Plot individual trials in light gray
        for trial_idx in range(len(high_indices)):
            ax_high.plot(high_data[trial_idx, :], alpha=0.1, color='gray', linewidth=0.5)
        
        # Plot mean in black
        mean_high = np.mean(high_data, axis=0)
        ax_high.plot(mean_high, 'k', linewidth=2, label='Mean')
        ax_high.set_title(f'Session {session+1} - High Tone (n={len(high_indices)} trials)')
        ax_high.set_xlabel('Time (samples)')
        ax_high.set_ylabel('Amplitude (mV)')
        ax_high.legend()
        ax_high.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_tone_responses(DATA, toneIndices, numSessions, time_ms, stimOnset, stimOffset):
    """
    Compare low vs high tone responses side by side for each session
    Original code from notebook Cell 20
    """
    # Original code from Cell 20
    # Compare low vs high tone responses side by side for each session
    fig, axes = plt.subplots(numSessions, 1, figsize=(14, 10))
    fig.suptitle('Low vs High Tone Comparison (Mean ± SEM)', fontsize=16, y=0.995)

    for session in range(numSessions):
        ax = axes[session]
        
        # Low tone
        low_indices = toneIndices[session]['low']
        low_data = DATA[session, 0][low_indices, :]
        mean_low = np.mean(low_data, axis=0)
        sem_low = np.std(low_data, axis=0) / np.sqrt(len(low_indices))
        
        # High tone
        high_indices = toneIndices[session]['high']
        high_data = DATA[session, 0][high_indices, :]
        mean_high = np.mean(high_data, axis=0)
        sem_high = np.std(high_data, axis=0) / np.sqrt(len(high_indices))
        
        # Plot both conditions
        ax.fill_between(time_ms, mean_low - sem_low, mean_low + sem_low, 
                        alpha=0.2, color='blue')
        ax.plot(time_ms, mean_low, 'b-', linewidth=2, label=f'Low Tone (n={len(low_indices)})')
        
        ax.fill_between(time_ms, mean_high - sem_high, mean_high + sem_high, 
                        alpha=0.2, color='red')
        ax.plot(time_ms, mean_high, 'r-', linewidth=2, label=f'High Tone (n={len(high_indices)})')
        
        # Mark stimulus period
        ax.axvspan(stimOnset, stimOffset, alpha=0.1, color='gray', label='Stimulus')
        
        ax.set_title(f'Session {session+1}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (mV)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_high_freq_power(signal, fs, freq_min, freq_max):
    """
    Calculate power in high-frequency band using Welch's method
    Original code from notebook Cell 23
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        freq_min: Minimum frequency for high-frequency band (Hz)
        freq_max: Maximum frequency (cutoff frequency)
        
    Returns:
        high_freq_power: Total power in high-frequency band
    """
    # Original code from Cell 23
    f, psd = scipy.signal.welch(signal, fs=fs, nperseg=256, noverlap=128)
    # Find frequency indices in the specified range
    freq_mask = (f >= freq_min) & (f <= freq_max)
    # Calculate total power in high-frequency band
    high_freq_power = np.sum(psd[freq_mask])
    return high_freq_power


def detect_outliers(DATA, toneIndices, numSessions, fs, high_freq_min=500, high_freq_max=1000):
    """
    Detect outliers using high-frequency power distribution (IQR-based)
    Original code from notebook Cell 23
    
    Args:
        DATA: Data array
        toneIndices: Dictionary with tone indices
        numSessions: Number of sessions
        fs: Sampling frequency
        high_freq_min: Minimum frequency for high-frequency band (Hz)
        high_freq_max: Maximum frequency (cutoff frequency)
        
    Returns:
        high_freq_powers: Dictionary storing high-frequency powers
        outlier_flags: Dictionary storing outlier flags
    """
    # Original code from Cell 23
    # Define high-frequency band for noise detection
    # High-frequency noise typically appears above 500 Hz
    
    # Calculate high-frequency power for each trial
    # Store high-frequency power for each session/tone combination
    high_freq_powers = {}
    outlier_flags = {}

    for session in range(numSessions):
        high_freq_powers[session] = {}
        outlier_flags[session] = {}
        
        for tone_type in ['low', 'high']:
            tone_indices = toneIndices[session][tone_type]
            trial_data = DATA[session, 0][tone_indices, :]
            
            # Calculate high-frequency power for each trial
            powers = []
            for trial_idx in range(len(tone_indices)):
                power = calculate_high_freq_power(trial_data[trial_idx, :], fs, 
                                                   high_freq_min, high_freq_max)
                powers.append(power)
            
            powers = np.array(powers)
            high_freq_powers[session][tone_type] = powers
            
            # Calculate IQR
            Q1 = np.percentile(powers, 25)
            Q3 = np.percentile(powers, 75)
            IQR = Q3 - Q1
            
            # Define outlier threshold (beyond Q3 + 1.5*IQR)
            outlier_threshold = Q3 + 1.5 * IQR
            
            # Identify outliers
            is_outlier = powers > outlier_threshold
            outlier_flags[session][tone_type] = is_outlier
            
            num_outliers = np.sum(is_outlier)
            num_valid = len(tone_indices) - num_outliers
    
    return high_freq_powers, outlier_flags


def remove_outliers(DATA, toneIndices, outlier_flags, numSessions):
    """
    Create cleaned data by removing outliers
    Original code from notebook Cell 24
    
    Args:
        DATA: Data array
        toneIndices: Dictionary with tone indices
        outlier_flags: Dictionary with outlier flags
        numSessions: Number of sessions
        
    Returns:
        rawData: Dictionary storing cleaned raw data
        validToneIndices: Dictionary storing valid tone indices after outlier removal
        trialsBeforeRejection: Dictionary storing trial counts before rejection
        trialsAfterRejection: Dictionary storing trial counts after rejection
    """
    # Original code from Cell 24
    # Create cleaned data by removing outliers
    rawData = {}  # Store cleaned raw data
    validToneIndices = {}  # Store valid tone indices after outlier removal

    trialsBeforeRejection = {}
    trialsAfterRejection = {}

    for session in range(numSessions):
        rawData[session] = {}
        validToneIndices[session] = {}
        trialsBeforeRejection[session] = {}
        trialsAfterRejection[session] = {}
        
        for tone_type in ['low', 'high']:
            original_indices = toneIndices[session][tone_type]
            is_outlier = outlier_flags[session][tone_type]
            
            # Get valid (non-outlier) indices
            valid_mask = ~is_outlier
            valid_indices = original_indices[valid_mask]
            
            # Store cleaned data
            rawData[session][tone_type] = DATA[session, 0][valid_indices, :]
            validToneIndices[session][tone_type] = valid_indices
            
            # Record trial counts
            trialsBeforeRejection[session][tone_type] = len(original_indices)
            trialsAfterRejection[session][tone_type] = len(valid_indices)

    # Also create combined valid indices for each session (all tones together)
    for session in range(numSessions):
        valid_low = validToneIndices[session]['low']
        valid_high = validToneIndices[session]['high']
        validToneIndices[session]['all'] = np.sort(np.concatenate([valid_low, valid_high]))
    
    return rawData, validToneIndices, trialsBeforeRejection, trialsAfterRejection


def print_rejection_summary(trialsBeforeRejection, trialsAfterRejection, numSessions):
    """
    Summary report: Trials before and after rejection
    Original code from notebook Cell 25
    """
    # Original code from Cell 25
    # Summary report: Trials before and after rejection

    # Per session
    for session in range(numSessions):
        total_before = (trialsBeforeRejection[session]['low'] + 
                        trialsBeforeRejection[session]['high'])
        total_after = (trialsAfterRejection[session]['low'] + 
                       trialsAfterRejection[session]['high'])

    # Per tone condition (across all sessions)
    for tone_type in ['low', 'high']:
        total_before = sum([trialsBeforeRejection[s][tone_type] for s in range(numSessions)])
        total_after = sum([trialsAfterRejection[s][tone_type] for s in range(numSessions)])

    # Overall
    total_before_all = sum([trialsBeforeRejection[s]['low'] + trialsBeforeRejection[s]['high'] 
                            for s in range(numSessions)])
    total_after_all = sum([trialsAfterRejection[s]['low'] + trialsAfterRejection[s]['high'] 
                           for s in range(numSessions)])
    print("\nOverall (All Sessions, All Tones):")
    print("-" * 70)
    print(f"  Before: {total_before_all} trials")
    print(f"  After: {total_after_all} trials")
    print(f"  Removed: {total_before_all - total_after_all} trials "
          f"({100*(total_before_all-total_after_all)/total_before_all:.1f}%)")
    print("=" * 70)


def visualize_cleaned_data(rawData, numSessions, time_ms, stimOnset, stimOffset, fs, dataSamples):
    """
    Visualize cleaned data (after outlier removal)
    Original code from notebook Cell 26
    """
    # Original code from Cell 26
    # Visualize cleaned data (after outlier removal)
    time_ms = np.arange(dataSamples) / fs * 1000

    fig, axes = plt.subplots(numSessions, 2, figsize=(16, 12))
    fig.suptitle('Cleaned LFP Data by Tone Condition (After Outlier Removal)', 
                 fontsize=16, y=0.995)

    for session in range(numSessions):
        for tone_idx, tone_type in enumerate(['low', 'high']):
            ax = axes[session, tone_idx]
            cleaned_data = rawData[session][tone_type]
            
            # Calculate mean and SEM
            mean_signal = np.mean(cleaned_data, axis=0)
            sem_signal = np.std(cleaned_data, axis=0) / np.sqrt(cleaned_data.shape[0])
            
            # Plot mean with SEM
            color = 'blue' if tone_type == 'low' else 'red'
            ax.fill_between(time_ms, mean_signal - sem_signal, mean_signal + sem_signal, 
                            alpha=0.3, color=color)
            ax.plot(time_ms, mean_signal, color=color, linewidth=2, label='Mean ± SEM')
            
            # Mark stimulus period
            ax.axvspan(stimOnset, stimOffset, alpha=0.2, color='gray', label='Stimulus')
            
            ax.set_title(f'Session {session+1} - {tone_type.upper()} Tone '
                        f'(n={cleaned_data.shape[0]} trials)')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (mV)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def store_outlier_data(DATA, toneIndices, outlier_flags, numSessions, dataSamples):
    """
    Store outlier indices and data for later use
    Original code from notebook Cell 27
    """
    # Original code from Cell 27
    # Store outlier indices for later use
    outlierIndices = {}
    for session in range(numSessions):
        outlierIndices[session] = {}
        for tone_type in ['low', 'high']:
            original_indices = toneIndices[session][tone_type]
            is_outlier = outlier_flags[session][tone_type]
            outlierIndices[session][tone_type] = original_indices[is_outlier]
            
    # Extract and store outlier data for visualization
    outlierData = {}
    for session in range(numSessions):
        outlierData[session] = {}
        for tone_type in ['low', 'high']:
            outlier_indices = outlierIndices[session][tone_type]
            if len(outlier_indices) > 0:
                outlierData[session][tone_type] = DATA[session, 0][outlier_indices, :]
            else:
                outlierData[session][tone_type] = np.array([]).reshape(0, dataSamples)
    
    return outlierIndices, outlierData


def visualize_outliers(outlierData, rawData, numSessions, time_ms, stimOnset, stimOffset):
    """
    Detailed visualization: Show all outlier trials separately
    Original code from notebook Cell 29
    """
    # Original code from Cell 29
    # Detailed visualization: Show all outlier trials separately
    fig, axes = plt.subplots(numSessions, 2, figsize=(18, 14))
    fig.suptitle('Removed Outlier Trials (Individual Traces)', fontsize=16, y=0.995)

    for session in range(numSessions):
        for tone_idx, tone_type in enumerate(['low', 'high']):
            ax = axes[session, tone_idx]
            
            outlier_data = outlierData[session][tone_type]
            
            if outlier_data.shape[0] > 0:
                # Plot all outlier trials
                for trial_idx in range(outlier_data.shape[0]):
                    ax.plot(time_ms, outlier_data[trial_idx, :], 
                           color='red', alpha=0.7, linewidth=1.5)
                
                # Plot mean of outliers
                mean_outlier = np.mean(outlier_data, axis=0)
                ax.plot(time_ms, mean_outlier, color='darkred', linewidth=3, 
                       linestyle='--', label='Mean of Outliers')
                
                # For comparison, plot mean of valid data
                valid_data = rawData[session][tone_type]
                if valid_data.shape[0] > 0:
                    mean_valid = np.mean(valid_data, axis=0)
                    ax.plot(time_ms, mean_valid, color='blue', linewidth=2, 
                           linestyle='-', label='Mean of Valid Data', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No outliers removed', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='gray')
            
            # Mark stimulus period
            ax.axvspan(stimOnset, stimOffset, alpha=0.15, color='gray')
            
            num_outliers = outlier_data.shape[0]
            ax.set_title(f'Session {session+1} - {tone_type.upper()} Tone '
                        f'({num_outliers} outliers removed)')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (mV)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

