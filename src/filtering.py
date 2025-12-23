"""
Filtering Module
Original code from midterm notebook - Cells 31, 32
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def setup_filtering_parameters():
    """
    Setup filtering parameters
    Original code from notebook Cell 31
    """
    # Original code from Cell 31
    windLength = 256
    wind = np.hanning(windLength)  # Hanning window (MATLAB: hanning(256))
    overl = 255  # Maximal overlap (MATLAB: overl = 255)
    binWidth = 5  # 5 Hz bands (MATLAB: binWidth = 5)
    maxFreq = 200  # Plot signal power up to 200 Hz (MATLAB: maxFreq = 200)
    
    return windLength, wind, overl, binWidth, maxFreq


def apply_filter(rawData, numSessions, cutoffFrequency, nyquistFrequency):
    """
    Apply 10th-order Butterworth low-pass filter
    Original code from notebook Cell 32
    
    Args:
        rawData: Dictionary with cleaned raw data
        numSessions: Number of sessions
        cutoffFrequency: Cutoff frequency
        nyquistFrequency: Nyquist frequency
        
    Returns:
        filteredData: List of dictionaries with filtered data
    """
    # Original code from Cell 32
    B, A = scipy.signal.butter(10, cutoffFrequency / nyquistFrequency, 'low')
    filteredData = []
    for session in range(numSessions):
        sessionData = rawData[session]
        filteredSession = {}
        for tone_type in ['low', 'high']:
            data = sessionData[tone_type]  # shape (n_trials, n_samples)
            filtered_trials = np.array([
                scipy.signal.filtfilt(B, A, trial) for trial in data
            ])
            filteredSession[tone_type] = filtered_trials
        filteredData.append(filteredSession)
    
    return filteredData


def visualize_filtering(rawData, filteredData, numSessions, time_ms, stimOnset, stimOffset, 
                       fs, windLength, wind, overl, maxFreq):
    """
    Visualization: plot mean waveform before and after filtering, with spectrograms of the mean
    Original code from notebook Cell 32
    """
    # Original code from Cell 32
    # Visualization: plot mean waveform before and after filtering, with spectrograms of the mean
    fig, axes = plt.subplots(numSessions, 4, figsize=(20, 4 * numSessions), sharex='col')
    if numSessions == 1:
        axes = axes[np.newaxis, :]

    for session in range(numSessions):
        for i, tone_type in enumerate(['low', 'high']):
            raw_trials = rawData[session][tone_type]
            filtered_trials = filteredData[session][tone_type]
            n_trials = raw_trials.shape[0]
            if n_trials == 0:
                # Time trace
                axes[session, 2 * i].text(0.5, 0.5, 'No trial', ha='center', va='center', fontsize=12)
                # Spectrogram
                axes[session, 2 * i + 1].text(0.5, 0.5, 'No trial', ha='center', va='center', fontsize=12)
                continue
            # Calculate mean across trials
            mean_raw = np.mean(raw_trials, axis=0)
            mean_filtered = np.mean(filtered_trials, axis=0)
            # Plot time traces of means
            axes[session, 2 * i].plot(time_ms, mean_raw, color='gray', alpha=0.7, label='Raw Mean')
            axes[session, 2 * i].plot(time_ms, mean_filtered, color='green', label='Filtered Mean')
            axes[session, 2 * i].set_title(f"Session {session+1} - {tone_type.capitalize()} Tone\nMean Waveform")
            axes[session, 2 * i].set_xlabel("Time (ms)")
            axes[session, 2 * i].set_ylabel("Amplitude (mV)")
            axes[session, 2 * i].axvspan(stimOnset, stimOffset, alpha=0.15, color='gray')
            axes[session, 2 * i].legend()
            axes[session, 2 * i].grid(alpha=0.3)
            # Plot spectrogram of filtered mean
            # Calculate spectrogram with proper parameters
            f, t, Sxx = scipy.signal.spectrogram(
                mean_filtered, 
                fs=fs, 
                window=wind,
                nperseg=windLength,
                noverlap=overl,
                nfft=windLength * 2  # nfft should be at least nperseg
            )
            freq_mask = f <= maxFreq
            f_filtered = f[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]
            # Plot spectrogram (MATLAB uses 'jet' colormap)
            im = axes[session, 2 * i + 1].pcolormesh(
                t * 1000,  # Time in ms
                f_filtered, 
                10 * np.log10(Sxx_filtered + 1e-10),  # Convert to dB scale
                shading='gouraud', 
                cmap='jet'  # MATLAB colormap (viridis -> jet로 변경)
            )
            axes[session, 2 * i + 1].set_ylabel('Frequency (Hz)')
            axes[session, 2 * i + 1].set_xlabel('Time (ms)')
            axes[session, 2 * i + 1].set_title(f"Session {session+1} - {tone_type.capitalize()} Tone\nSpectrogram (Filtered Mean)")
            axes[session, 2 * i + 1].set_ylim([0, maxFreq])  # 1200 -> maxFreq (200)로 변경

    plt.tight_layout()
    plt.show()

