# SAR Rice Mapping with Spatiotemporal Coupling Models

This repository contains the source code used in the study:

**Effects of Spatiotemporal Model Coupling Strategies on the Mapping Accuracy and Uncertainty of SAR-Derived Rice Paddies**

The study systematically compares different spatiotemporal modeling strategies for rice mapping using Sentinel-1 time-series SAR data.

## Model Categories

Three groups of deep learning models are included.

### 1. Temporal-first Models

Temporal features are first extracted using LSTM or Transformer and then processed by spatial segmentation networks.

Models include:

- LSTM + UNet
- LSTM + DeepLab
- LSTM + SegNet
- Transformer + UNet
- Transformer + DeepLab
- Transformer + SegNet

The corresponding scripts are provided in:

```text
temporal-first/
```

### 2. Spatial-first Models

Spatial features are first extracted using 2D CNNs and then processed by LSTM or Transformer for temporal modeling.

Models include:

- UNet + LSTM
- DeepLab + LSTM
- SegNet + LSTM
- UNet + Transformer
- DeepLab + Transformer
- SegNet + Transformer

The corresponding scripts are provided in:

```text
spatial-first/
```

### 3. Baseline Models

The baseline models include standalone temporal models, 2D CNNs, 3D CNNs, and ConvLSTM.

Models include:

- LSTM
- Transformer
- 2D-UNet
- 2D-DeepLab
- 2D-SegNet
- 3D-UNet
- 3D-DeepLab
- 3D-SegNet
- ConvLSTM

The corresponding scripts are provided in:

```text
baseline/
```

## Generalization Experiments

The models were evaluated under four generalization scenarios:

- Within-region spatial generalization
- Cross-regional spatial generalization
- Within-region temporal generalization
- Cross-regional spatiotemporal generalization

Arkansas was used as the primary study region, while Colusa County, California, was used as an independent external test region.

## Data

The study uses:

- Sentinel-1 SAR time-series imagery
- USDA Cropland Data Layer (CDL)

For each year, nine temporal intervals were constructed using VV and VH polarization data, resulting in 18 input bands.

The raw remote sensing datasets are not included in this repository.

## Environment

The main Python packages used in the experiments include:

```text
Python
TensorFlow / Keras
NumPy
Rasterio
Matplotlib
scikit-learn
```

The exact environment may vary depending on the model and experiment.

## Repository Structure

```text
SAR-rice-mapping-spatiotemporal-coupling/
├── temporal-first/
├── spatial-first/
├── baseline/
└── README.md
```

## Notes

Users should modify the dataset and output paths in each script according to their local environment before running the experiments.

Large remote sensing datasets, trained model weights, and experiment outputs are not included in this repository.

## Contact

For questions regarding the code or experiments, please contact the authors of the corresponding paper.
