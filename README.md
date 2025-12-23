# BRI519 Final Project - Mouse LFP Data Analysis

## Project Overview

This project performs comprehensive analysis of Local Field Potential (LFP) data from mouse experiments. The analysis pipeline includes data loading, outlier detection and removal, signal filtering, and two different analytical methods for examining neural responses to different tone conditions across multiple sessions.

## Features

- **Data Loading**: Load and inspect MATLAB data files (.mat format)
- **Outlier Detection**: High-frequency power-based outlier detection using IQR method
- **Signal Filtering**: 10th-order Butterworth low-pass filter
- **Analysis Method 1**: Peri-Stimulus Time Analysis (PSTH-like approach)
- **Analysis Method 2**: Cross-Session and Cross-Condition Correlation Analysis
- **Combined Analysis**: Analysis across all sessions combined
- **Visualization**: Comprehensive plotting of results in both time and frequency domains

## Project Structure

```
bri519-final-project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── main.py                  # Main execution script
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── constants.py         # Constants and configuration
│   ├── data_loader.py       # Data loading functions
│   ├── outlier_detection.py # Outlier detection and removal
│   ├── filtering.py         # Signal filtering
│   ├── analysis.py          # Main analysis methods
│   └── data_saver.py        # Results saving
├── data/                    # Data directory (place mouseLFP.mat here)
├── results/                 # Output directory for results
└── notebooks/              # Jupyter notebooks
    └── tutorial.ipynb      # Tutorial notebook with step-by-step examples
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data file (`mouseLFP.mat`) in the `data/` directory

## Usage

### Basic Execution

Run the main analysis script:

```bash
python main.py
```

This will execute the complete analysis pipeline:
1. Load constants and configuration
2. Load the dataset from `data/mouseLFP.mat`
3. Detect and remove outliers
4. Apply low-pass filtering
5. Perform Method 1 (PSTH-like) analysis
6. Perform Method 2 (Correlation) analysis
7. Analyze combined sessions
8. Save results to `results/` directory

### Output Files

The analysis generates the following output files in the `results/` directory:

- `analysis_results.pkl`: Pickle file containing all analysis results and parameters
- `analysis_results.npz`: NumPy compressed archive with key data arrays

### Tutorial Notebook

A comprehensive tutorial notebook is available in `notebooks/tutorial.ipynb` that demonstrates:

- Step-by-step usage of each module
- How to run the complete analysis pipeline
- Visualization examples
- Detailed explanations of each analysis method

To use the tutorial:

1. Install Jupyter Notebook (if not already installed):
```bash
pip install jupyter
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/tutorial.ipynb
```

3. Follow the tutorial cells sequentially to learn how to use each module

### Module Usage

You can also import and use individual modules:

```python
from src.data_loader import load_data
from src.outlier_detection import detect_outliers
from src.filtering import apply_filter
from src.analysis import method1_psth_analysis, method2_correlation_analysis

# Load data
DATA, dataSamples = load_data('data/mouseLFP.mat')

# Your analysis code here...
```

## Analysis Methods

### Method 1: Peri-Stimulus Time Analysis (PSTH-like)

This method analyzes LFP responses in the time domain, similar to Peri-Stimulus Time Histogram (PSTH) analysis for spike trains. It calculates:
- Mean and SEM of LFP responses
- Baseline activity
- Peak amplitude and latency
- Response duration

### Method 2: Cross-Session and Cross-Condition Correlation Analysis

This method examines correlations:
- Between sessions for each tone condition
- Between low and high tone conditions within each session
- Frequency domain correlations

## Parameters

Key parameters (defined in `src/constants.py`):

- `cutoffFrequency`: 1000 Hz (low-pass filter cutoff)
- `fs`: 10000 Hz (sampling frequency)
- `stimOnset`: 100 ms (stimulus onset time)
- `stimOffset`: 150 ms (stimulus offset time)
- `numSessions`: 4 (number of experimental sessions)

## Code Structure

The code is organized into modular components:

- **constants.py**: All constant values and configuration
- **data_loader.py**: Functions for loading and inspecting data
- **outlier_detection.py**: Outlier detection using high-frequency power analysis
- **filtering.py**: Butterworth low-pass filter implementation
- **analysis.py**: Main analysis methods (Method 1 and Method 2)
- **data_saver.py**: Functions for saving analysis results

All original code logic and comments from the midterm notebook have been preserved in the modular structure.

## Requirements

- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

## Docker Usage

This project is containerized using Docker for reproducible execution.

### Docker Hub Image

The Docker image is available on Docker Hub:
- **Image**: `hobacteria/bri519-final-project:latest`
- **Docker Hub URL**: https://hub.docker.com/r/hobacteria/bri519-final-project

### Pull and Run from Docker Hub

```bash
# Pull the image from Docker Hub
docker pull hobacteria/bri519-final-project:latest

# Run the container with data volume mount
docker run --rm \
  -v /path/to/your/data:/app/data \
  -v /path/to/your/results:/app/results \
  hobacteria/bri519-final-project:latest
```

**Note**: Replace `/path/to/your/data` with the path to your directory containing `mouseLFP.mat`, and `/path/to/your/results` with the path where you want to save the results.

### Build Docker Image Locally

If you want to build the image from source:

```bash
# Clone the repository
git clone https://github.com/hobacteria/bri519-final.git
cd bri519-final

# Build the Docker image
docker build -t bri519-final-project:latest .

# Run the container
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  bri519-final-project:latest
```
**Note**: `mouseLFP.mat` is not included in the Docker image. Please mount the directory containing `mouseLFP.mat`

### Verify Docker Deployment

To verify that the Docker Hub image works identically to local execution:

```bash
# Pull and run from Docker Hub
docker pull hobacteria/bri519-final-project:latest
docker run --rm \
  -v $(pwd)/path/to/your/data:/app/data \
  -v $(pwd)/path/to/your/results:/app/results \
  hobacteria/bri519-final-project:latest

# Compare results with local execution
# The output should be identical
```


## Course

BRI519 (Fall 2025) - Final Assignment

