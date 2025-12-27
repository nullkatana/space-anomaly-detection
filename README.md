# Space Anomaly Detection

Deep learning system for detecting rare astronomical events and anomalies in observatory images using Variational Autoencoders (VAE). Identifies transients, gravitational lensing, supernovae, and unusual cosmic phenomena in telescope data.

## Project Overview

This project implements an unsupervised deep learning pipeline to automatically flag anomalous events in astronomical imagery. By training on "normal" cosmic objects, the model learns to identify unusual or rare phenomena that warrant further investigation by astronomers.

### Key Features

- **Unsupervised Learning**: Detects anomalies without requiring labeled rare events
- **VAE Architecture**: Probabilistic latent space for robust anomaly scoring
- **FITS Support**: Native handling of astronomical image formats
- **Heatmap Visualization**: Localized anomaly detection with confidence scores
- **Multi-scale Processing**: Handles images from various observatories and wavelengths

## Architecture

```
Observatory Image (FITS)
         ↓
   Preprocessing
   (normalize, augment, patch extraction)
         ↓
Variational Autoencoder (VAE)
   Encoder → Latent Space → Decoder
         ↓
   Anomaly Scoring
   (reconstruction error + KL divergence)
         ↓
Flagged Anomalies + Heatmaps
```

## Datasets

Currently supporting:

- **Hubble Legacy Archive**: High-quality deep space imagery
- **Zwicky Transient Facility (ZTF)**: Real-time transient detection data
- Custom FITS image collections

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~10GB storage for datasets

### Installation

```bash
# Clone repository
git clone https://github.com/nullkatana/space-anomaly-detection.git
cd space-anomaly-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Coming soon: Example usage for training and detection
```

## Project Structure
```
space-anomaly-detection/
├── data/
│   ├── raw/              # Original FITS files
│   ├── processed/        # Preprocessed image patches
│   └── annotations/      # Known anomalies (optional)
├── src/
│   ├── __init__.py       # Package initialization
│   ├── preprocessing.py  # FITS handling and augmentation
│   ├── models.py         # VAE architecture
│   ├── train.py          # Training pipeline
│   ├── detect.py         # Inference and anomaly scoring
│   └── visualize.py      # Heatmap generation
├── notebooks/            # Jupyter notebooks for exploration
├── models/               # Saved model weights
├── results/              # Detection outputs and metrics
│   ├── anomalies/        # Flagged detections
│   └── metrics/          # ROC curves, performance metrics
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Methodology

### Preprocessing
- Log scaling for astronomical dynamic range
- Percentile clipping for outlier removal
- Rotation and flip augmentation (space has no orientation)
- 128×128 or 256×256 patch extraction

### Model
- **Encoder**: 3-layer CNN with batch normalization
- **Latent Space**: 128-256 dimensions, probabilistic
- **Decoder**: Transposed convolutions for reconstruction
- **Loss**: Reconstruction error + KL divergence

### Anomaly Detection
Combined scoring:
- 70% reconstruction error (MSE)
- 30% Mahalanobis distance in latent space
- Threshold tuning via ROC curve analysis

## Performance Metrics

- ROC-AUC score
- Precision-Recall curves
- False positive rate
- Detection latency
- Reconstruction error distribution

## Roadmap

- [x] Project structure and environment setup
- [ ] Dataset acquisition and exploration
- [ ] Preprocessing pipeline implementation
- [ ] VAE model architecture
- [ ] Training loop with validation
- [ ] Anomaly detection and scoring
- [ ] Visualization and reporting
- [ ] Real-time inference pipeline
- [ ] Multi-wavelength data fusion
- [ ] Attention mechanism for interpretability

## License

MIT License - See [LICENSE](LICENSE) file for details.

## References

- Hubble Legacy Archive: https://hla.stsci.edu/
- Zwicky Transient Facility: https://www.ztf.caltech.edu/
- VAE paper: Kingma & Welling (2013)
- Astronomical data processing: Astropy Project

---

**Author:** Yooh Brito  
**Date:** December 2025
**Status:** v0.1 - Environment Setup Complete