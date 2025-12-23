"""
Analysis Module - Method 1 and Method 2
Original code from midterm notebook - Cells 37, 38, 39, 45, 46, 49
"""

import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt


def method1_psth_analysis(filteredData, numSessions, stimOnset, stimOffset, fs):
    """
    Method 1: Peri-Stimulus Time Analysis (PSTH-like approach for LFP)
    Original code from notebook Cell 37
    
    Args:
        filteredData: List of dictionaries with filtered data
        numSessions: Number of sessions
        stimOnset: Stimulus onset time (ms)
        stimOffset: Stimulus offset time (ms)
        fs: Sampling frequency
        
    Returns:
        meanLFPs: Dictionary with mean LFP responses
        semLFPs: Dictionary with SEM LFP responses
        responseMetrics: Dictionary with response metrics
    """
    # Original code from Cell 37
    # Method 1: Peri-Stimulus Time Analysis (PSTH-like approach for LFP)
    # Inspired by Chapter 3: Wrangling Spike Trains

    print("=" * 70)
    print("Method 1: Peri-Stimulus Time Analysis")
    print("=" * 70)

    # Calculate mean LFP responses for each session and tone condition
    # Similar to calculating mean spike count in PSTH
    meanLFPs = {}
    semLFPs = {}
    responseMetrics = {}

    for session in range(numSessions):
        meanLFPs[session] = {}
        semLFPs[session] = {}
        responseMetrics[session] = {}
        
        for tone_type in ['low', 'high']:
            # Get filtered data (after outlier removal)
            filtered_trials = filteredData[session][tone_type]
            
            # Calculate mean and SEM (similar to PSTH mean calculation)
            meanLFPs[session][tone_type] = np.mean(filtered_trials, axis=0)
            semLFPs[session][tone_type] = np.std(filtered_trials, axis=0) / np.sqrt(filtered_trials.shape[0])
            
            # Quantify response characteristics
            mean_signal = meanLFPs[session][tone_type]
            stim_start_idx = int(stimOnset * fs / 1000)  # Convert ms to sample index
            stim_end_idx = int(stimOffset * fs / 1000)
            
            # Baseline: mean before stimulus
            baseline = np.mean(mean_signal[:stim_start_idx])
            baseline_std = np.std(mean_signal[:stim_start_idx])
            
            # Response window: during and after stimulus
            response_window = mean_signal[stim_start_idx:]
            
            # Peak amplitude (maximum deviation from baseline)
            peak_amplitude = np.max(np.abs(response_window - baseline))
            peak_idx = np.argmax(np.abs(response_window - baseline)) + stim_start_idx
            peak_latency = (peak_idx - stim_start_idx) * 1000 / fs  # ms after stimulus onset
            
            # Response duration (time above baseline + 2*std)
            threshold = baseline + 2 * baseline_std
            above_threshold = np.abs(response_window) > np.abs(threshold)
            if np.any(above_threshold):
                response_duration = np.sum(above_threshold) * 1000 / fs  # ms
            else:
                response_duration = 0
            
            responseMetrics[session][tone_type] = {
                'baseline': baseline,
                'peak_amplitude': peak_amplitude,
                'peak_latency': peak_latency,
                'response_duration': response_duration
            }
            
            print(f"\nSession {session+1} - {tone_type.upper()} Tone:")
            print(f"  Baseline: {baseline:.4f} mV")
            print(f"  Peak amplitude: {peak_amplitude:.4f} mV")
            print(f"  Peak latency: {peak_latency:.2f} ms after stimulus onset")
            print(f"  Response duration: {response_duration:.2f} ms")

    print("\n" + "=" * 70)
    
    return meanLFPs, semLFPs, responseMetrics


def visualize_method1_time(meanLFPs, semLFPs, responseMetrics, numSessions, time_ms, 
                           stimOnset, stimOffset, fs):
    """
    Visualize Method 1: Time domain analysis (PSTH-like)
    Original code from notebook Cell 38
    """
    # Original code from Cell 38
    # Visualize Method 1: Time domain analysis (PSTH-like)
    time_ms = np.arange(len(meanLFPs[0]['low'])) / fs * 1000

    fig, axes = plt.subplots(numSessions, 2, figsize=(16, 4 * numSessions))
    fig.suptitle('Method 1: Peri-Stimulus Time Analysis (PSTH-like for LFP)', 
                 fontsize=16, y=0.995)

    for session in range(numSessions):
        for tone_idx, tone_type in enumerate(['low', 'high']):
            ax = axes[session, tone_idx]
            
            mean_signal = meanLFPs[session][tone_type]
            sem_signal = semLFPs[session][tone_type]
            
            # Plot mean ± SEM (similar to PSTH with error bars)
            ax.fill_between(time_ms, mean_signal - sem_signal, mean_signal + sem_signal, 
                           alpha=0.3, color='blue' if tone_type == 'low' else 'red', 
                           label='±SEM')
            ax.plot(time_ms, mean_signal, color='blue' if tone_type == 'low' else 'red', 
                   linewidth=2, label='Mean LFP')
            
            # Mark baseline
            baseline = responseMetrics[session][tone_type]['baseline']
            ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
            
            # Mark stimulus period
            ax.axvspan(stimOnset, stimOffset, alpha=0.2, color='green', label='Stimulus')
            
            # Mark peak response
            peak_latency = responseMetrics[session][tone_type]['peak_latency']
            peak_time = stimOnset + peak_latency
            peak_amplitude = responseMetrics[session][tone_type]['peak_amplitude']
            ax.plot(peak_time, mean_signal[int(peak_time * fs / 1000)], 
                   'o', color='orange', markersize=10, label='Peak Response')
            
            ax.set_title(f'Session {session+1} - {tone_type.upper()} Tone\n'
                        f'Peak: {peak_amplitude:.4f} mV at {peak_latency:.1f} ms')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (mV)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_method1_frequency(meanLFPs, numSessions, fs):
    """
    Frequency domain analysis for Method 1
    Calculate power spectrum for mean LFP responses
    Original code from notebook Cell 39
    """
    # Original code from Cell 39
    # Frequency domain analysis for Method 1
    # Calculate power spectrum for mean LFP responses

    fig, axes = plt.subplots(numSessions, 2, figsize=(16, 4 * numSessions))
    fig.suptitle('Method 1: Frequency Domain Analysis (Power Spectrum)', 
                 fontsize=16, y=0.995)

    for session in range(numSessions):
        for tone_idx, tone_type in enumerate(['low', 'high']):
            ax = axes[session, tone_idx]
            
            mean_signal = meanLFPs[session][tone_type]
            
            # Calculate power spectrum using Welch's method
            f, psd = scipy.signal.welch(mean_signal, fs=fs, nperseg=256, noverlap=128)
            
            # Plot power spectrum (up to 200 Hz for LFP)
            freq_mask = f <= 200
            ax.plot(f[freq_mask], 10 * np.log10(psd[freq_mask] + 1e-10), 
                   color='blue' if tone_type == 'low' else 'red', linewidth=2)
            
            ax.set_title(f'Session {session+1} - {tone_type.upper()} Tone\nPower Spectrum')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (dB)')
            ax.set_xlim([0, 200])
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def method2_correlation_analysis(meanLFPs, numSessions):
    """
    Method 2: Cross-Session and Cross-Condition Correlation Analysis
    Original code from notebook Cell 45
    
    Args:
        meanLFPs: Dictionary with mean LFP responses
        numSessions: Number of sessions
        
    Returns:
        rSC_sessions: Dictionary with session correlation matrices
        rSC_conditions: Array with condition correlations within sessions
    """
    # Original code from Cell 45
    # Method 2: Cross-Session and Cross-Condition Correlation Analysis
    # Inspired by Chapter 4: Correlating Spike Trains

    print("=" * 70)
    print("Method 2: Cross-Session and Cross-Condition Correlation Analysis")
    print("=" * 70)

    # Calculate correlations between sessions for each tone condition
    # Similar to spike count correlation between channels
    rSC_sessions = {}  # Session correlation matrix
    rSC_conditions = {}  # Condition correlation within sessions

    for tone_type in ['low', 'high']:
        # Initialize correlation matrix for sessions (numSessions x numSessions)
        rSC_sessions[tone_type] = np.zeros((numSessions, numSessions))
        
        # Calculate pairwise correlations between sessions
        for row in range(numSessions):
            for col in range(numSessions):
                if row == col:
                    rSC_sessions[tone_type][row, col] = 1.0  # Self-correlation
                else:
                    # Get mean LFP responses
                    signal1 = meanLFPs[row][tone_type]
                    signal2 = meanLFPs[col][tone_type]
                    
                    # Calculate Pearson correlation (similar to Chapter 4)
                    r, _ = scipy.stats.pearsonr(signal1, signal2)
                    rSC_sessions[tone_type][row, col] = r
        
        print(f"\n{tone_type.upper()} Tone - Cross-Session Correlation Matrix:")
        print(rSC_sessions[tone_type])
        print(f"  Global Mean Correlation: {np.mean(rSC_sessions[tone_type][np.triu_indices(numSessions, k=1)]):.4f}")

    # Calculate correlations between low and high tone within each session
    rSC_conditions = np.zeros(numSessions)
    for session in range(numSessions):
        signal_low = meanLFPs[session]['low']
        signal_high = meanLFPs[session]['high']
        r, _ = scipy.stats.pearsonr(signal_low, signal_high)
        rSC_conditions[session] = r
        print(f"\nSession {session+1} - Low vs High Tone Correlation: {r:.4f}")

    print(f"\nOverall Mean Cross-Condition Correlation: {np.mean(rSC_conditions):.4f}")
    print("=" * 70)
    
    return rSC_sessions, rSC_conditions


def visualize_method2_correlation(rSC_sessions, rSC_conditions, numSessions):
    """
    Visualize Method 2: Correlation matrices (similar to Chapter 4)
    Original code from notebook Cell 46
    """
    # Original code from Cell 46
    # Visualize Method 2: Correlation matrices (similar to Chapter 4)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Method 2: Correlation Analysis Matrices', fontsize=16, y=1.02)

    # Plot cross-session correlation for low tone
    im1 = axes[0].imshow(rSC_sessions['low'], cmap='jet', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('Low Tone: Cross-Session Correlation')
    axes[0].set_xlabel('Session')
    axes[0].set_ylabel('Session')
    axes[0].set_xticks(range(numSessions))
    axes[0].set_yticks(range(numSessions))
    axes[0].set_xticklabels([f'S{i+1}' for i in range(numSessions)])
    axes[0].set_yticklabels([f'S{i+1}' for i in range(numSessions)])
    plt.colorbar(im1, ax=axes[0], label='Correlation')

    # Plot cross-session correlation for high tone
    im2 = axes[1].imshow(rSC_sessions['high'], cmap='jet', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('High Tone: Cross-Session Correlation')
    axes[1].set_xlabel('Session')
    axes[1].set_ylabel('Session')
    axes[1].set_xticks(range(numSessions))
    axes[1].set_yticks(range(numSessions))
    axes[1].set_xticklabels([f'S{i+1}' for i in range(numSessions)])
    axes[1].set_yticklabels([f'S{i+1}' for i in range(numSessions)])
    plt.colorbar(im2, ax=axes[1], label='Correlation')

    # Plot cross-condition correlation within sessions
    im3 = axes[2].bar(range(numSessions), rSC_conditions, color=['blue', 'green', 'orange', 'red'])
    axes[2].set_title('Low vs High Tone Correlation (Within Sessions)')
    axes[2].set_xlabel('Session')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xticks(range(numSessions))
    axes[2].set_xticklabels([f'Session {i+1}' for i in range(numSessions)])
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([-1, 1])

    plt.tight_layout()
    plt.show()


def combined_sessions_analysis(filteredData, numSessions, meanLFPs, rSC_conditions, 
                              stimOnset, stimOffset, fs):
    """
    Apply both methods to the LFP data combined across the four sessions
    Original code from notebook Cell 49
    
    Args:
        filteredData: List of dictionaries with filtered data
        numSessions: Number of sessions
        meanLFPs: Dictionary with mean LFP responses
        rSC_conditions: Array with condition correlations
        stimOnset: Stimulus onset time (ms)
        stimOffset: Stimulus offset time (ms)
        fs: Sampling frequency
        
    Returns:
        lfp_combined: Dictionary with combined LFP data
        r_combined_conditions: Combined cross-condition correlation
        r_combined_freq: Combined frequency domain correlation
    """
    # Original code from Cell 49
    # (iii) Apply both methods to the LFP data combined across the four sessions, present the results in both the time and frequency domains,
    # and discuss findings, including comparison with results from analyzing each session separately.

    # === COMBINE DATA ACROSS SESSIONS ===
    # filteredData structure: filteredData[session][tone_type] where each is (n_trials, n_samples)
    print("=" * 70)
    print("Combined Analysis Across All Sessions")
    print("=" * 70)

    # Combine trials and sessions for each condition
    lfp_combined = {'low': [], 'high': []}
    for sess in range(numSessions):
        # filteredData[sess] is a dictionary with 'low' and 'high' keys
        lfp_combined['low'].append(filteredData[sess]['low'])
        lfp_combined['high'].append(filteredData[sess]['high'])

    # Concatenate trials across all sessions per condition (shape: trials_total x nSamples)
    lfp_combined['low'] = np.concatenate(lfp_combined['low'], axis=0)
    lfp_combined['high'] = np.concatenate(lfp_combined['high'], axis=0)

    print(f"Combined Low Tone: {lfp_combined['low'].shape[0]} trials")
    print(f"Combined High Tone: {lfp_combined['high'].shape[0]} trials")

    # 1. TIME DOMAIN: Mean ± SEM LFP trace for each tone
    time_ms = np.arange(lfp_combined['low'].shape[1]) / fs * 1000  # ms

    # Method 1: Combined PSTH-like analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Method 1: Combined Sessions - Peri-Stimulus Time Analysis', fontsize=16)

    for tone_idx, (tone, color) in enumerate(zip(['low', 'high'], ['blue', 'red'])):
        ax = axes[tone_idx]
        mean_lfp = lfp_combined[tone].mean(axis=0)
        sem_lfp = lfp_combined[tone].std(axis=0) / np.sqrt(lfp_combined[tone].shape[0])
        
        ax.fill_between(time_ms, mean_lfp - sem_lfp, mean_lfp + sem_lfp, 
                       alpha=0.3, color=color, label='±SEM')
        ax.plot(time_ms, mean_lfp, color=color, linewidth=2, label='Mean LFP')
        
        # Mark stimulus period
        ax.axvspan(stimOnset, stimOffset, alpha=0.2, color='gray', label='Stimulus')
        
        # Mark baseline
        baseline = np.mean(mean_lfp[:int(stimOnset * fs / 1000)])
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        ax.set_title(f'{tone.capitalize()} Tone (n={lfp_combined[tone].shape[0]} trials)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (mV)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 2. FREQUENCY DOMAIN: Power Spectral Density for combined data
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Method 1: Combined Sessions - Frequency Domain Analysis', fontsize=16)

    for tone_idx, (tone, color) in enumerate(zip(['low', 'high'], ['blue', 'red'])):
        ax = axes[tone_idx]
        
        # Calculate power spectrum for each trial, then average
        f, Pxx = scipy.signal.welch(lfp_combined[tone], fs=fs, axis=1, nperseg=256, noverlap=128)
        
        # Average across trials
        mean_psd = np.mean(Pxx, axis=0)
        sem_psd = np.std(Pxx, axis=0) / np.sqrt(Pxx.shape[0])
        
        # Filter to 0-200 Hz
        freq_mask = f <= 200
        f_filtered = f[freq_mask]
        mean_psd_filtered = mean_psd[freq_mask]
        sem_psd_filtered = sem_psd[freq_mask]
        
        ax.fill_between(f_filtered, 
                        mean_psd_filtered - sem_psd_filtered,
                        mean_psd_filtered + sem_psd_filtered,
                        alpha=0.3, color=color)
        ax.plot(f_filtered, 10 * np.log10(mean_psd_filtered + 1e-10), 
               color=color, linewidth=2)
        
        ax.set_title(f'{tone.capitalize()} Tone Power Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_xlim([0, 200])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Method 2: Correlation analysis for combined data
    print("\n" + "=" * 70)
    print("Method 2: Combined Sessions - Correlation Analysis")
    print("=" * 70)

    # Calculate mean responses for combined data
    mean_combined_low = lfp_combined['low'].mean(axis=0)
    mean_combined_high = lfp_combined['high'].mean(axis=0)

    # Cross-condition correlation (low vs high tone in combined data)
    r_combined_conditions, _ = scipy.stats.pearsonr(mean_combined_low, mean_combined_high)
    print(f"\nCombined Sessions - Low vs High Tone Correlation: {r_combined_conditions:.4f}")

    # Compare with individual session correlations
    print("\nComparison with Individual Sessions:")
    for session in range(numSessions):
        r_session = rSC_conditions[session]
        print(f"  Session {session+1}: {r_session:.4f}")

    # Visualize combined correlation
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Method 2: Combined Sessions - Correlation Analysis', fontsize=16)

    # Plot combined mean responses
    axes[0].plot(time_ms, mean_combined_low, color='blue', linewidth=2, label='Low Tone')
    axes[0].plot(time_ms, mean_combined_high, color='red', linewidth=2, label='High Tone')
    axes[0].axvspan(stimOnset, stimOffset, alpha=0.2, color='gray', label='Stimulus')
    axes[0].set_title(f'Combined Mean Responses\nCorrelation: r = {r_combined_conditions:.4f}')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Compare individual vs combined correlations
    sessions = [f'S{i+1}' for i in range(numSessions)]
    individual_corrs = [rSC_conditions[i] for i in range(numSessions)]
    axes[1].bar(sessions, individual_corrs, color=['blue', 'green', 'orange', 'red'], 
               alpha=0.7, label='Individual Sessions')
    axes[1].axhline(y=r_combined_conditions, color='black', linestyle='--', 
                   linewidth=2, label=f'Combined: {r_combined_conditions:.4f}')
    axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[1].set_title('Cross-Condition Correlation Comparison')
    axes[1].set_ylabel('Correlation')
    axes[1].set_ylim([-1, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Frequency domain correlation for combined data
    f_combined_low, psd_combined_low = scipy.signal.welch(mean_combined_low, fs=fs, nperseg=256, noverlap=128)
    f_combined_high, psd_combined_high = scipy.signal.welch(mean_combined_high, fs=fs, nperseg=256, noverlap=128)

    freq_mask = (f_combined_low <= 200)
    r_combined_freq, _ = scipy.stats.pearsonr(psd_combined_low[freq_mask], psd_combined_high[freq_mask])
    print(f"\nCombined Sessions - Frequency Domain Correlation: {r_combined_freq:.4f}")

    # Note: rSC_freq_conditions would need to be calculated separately if needed
    # For now, we'll skip that comparison as it wasn't in the original code
    
    print("\n" + "=" * 70)
    
    return lfp_combined, r_combined_conditions, r_combined_freq

