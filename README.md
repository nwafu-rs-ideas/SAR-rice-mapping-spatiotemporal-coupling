# SAR Rice Mapping with Spatiotemporal Coupling Models

This repository contains the source code and supporting data used in the study:

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

`temporal-first/`

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

`spatial-first/`

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

`baseline/`

## Generalization Experiments

The models were evaluated under four generalization scenarios:

- Within-region spatial generalization
- Cross-regional spatial generalization
- Within-region temporal generalization
- Cross-regional spatiotemporal generalization

Arkansas was used as the primary study region, while Colusa County, California, was used as an independent external test region.

## Data

The study uses Sentinel-1 SAR time-series imagery and USDA Cropland Data Layer (CDL) data.

For each year, nine temporal intervals were constructed using VV and VH polarization data, resulting in 18 input bands.

### 1. Sentinel-1 SAR Data

The Sentinel-1 SAR imagery used in this study was obtained through the Google Earth Engine (GEE) platform.

The original Sentinel-1 SAR imagery is publicly accessible through the GEE data catalog:

https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD

The original SAR imagery is not directly included in this repository due to its large data volume.

The vector boundaries provided in the `data/` directory define the spatial extents of the Arkansas and Colusa study areas. These boundaries can be used to identify the geographic coverage of the study areas and retrieve corresponding Sentinel-1 imagery through GEE.

### 2. Cropland Data Layer (CDL)

The CDL data used in this study were obtained from the United States Department of Agriculture (USDA) National Agricultural Statistics Service (NASS).

The original CDL datasets are publicly accessible at:

https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php

The `data/` directory contains the CDL data for the two study areas.

**Arkansas:**

- `AR_CDL_2017_Clip.tif`
- `AR_CDL_2018_Clip.tif`
- `AR_CDL_2019_Clip.tif`
- `AR_CDL_2021_Clip.tif`

**Colusa County, California:**

- `D_CDL_NASS_DATA_CACHE_extract_3_CDL_2021_06011-Colusa.tif`

These CDL datasets were used to generate reference labels for rice mapping.

### 3. Study Area Boundaries

The `data/` directory also contains vector boundaries defining the spatial extents of the two study areas.

**Arkansas:**

- `Arkansas_Study_Area.shp`
- `Arkansas_Study_Area.shx`
- `Arkansas_Study_Area.dbf`
- `Arkansas_Study_Area.prj`
- `Arkansas_Study_Area.cpg`

**Colusa County, California:**

- `Colusa_Study_Area.shp`
- `Colusa_Study_Area.shx`
- `Colusa_Study_Area.dbf`
- `Colusa_Study_Area.prj`
- `Colusa_Study_Area.cpg`

The vector boundaries were generated from the valid spatial coverage of the remote sensing imagery used in this study.

These files can be used to identify the study areas and support the retrieval of corresponding remote sensing data through GEE.

### 4. Data Availability

The CDL reference data and vector boundaries of the Arkansas and Colusa study areas are provided in the `data/` directory.

The original Sentinel-1 SAR imagery can be obtained through Google Earth Engine.

The source code for the evaluated models is also publicly available in this repository.

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
├── data/
│   ├── Arkansas/
│   │   ├── AR_CDL_2017_Clip.tif
│   │   ├── AR_CDL_2018_Clip.tif
│   │   ├── AR_CDL_2019_Clip.tif
│   │   ├── AR_CDL_2021_Clip.tif
│   │   ├── Arkansas_Study_Area.shp
│   │   ├── Arkansas_Study_Area.shx
│   │   ├── Arkansas_Study_Area.dbf
│   │   ├── Arkansas_Study_Area.prj
│   │   └── Arkansas_Study_Area.cpg
│   └── Colusa/
│       ├── D_CDL_NASS_DATA_CACHE_extract_3_CDL_2021_06011-Colusa.tif
│       ├── Colusa_Study_Area.shp
│       ├── Colusa_Study_Area.shx
│       ├── Colusa_Study_Area.dbf
│       ├── Colusa_Study_Area.prj
│       └── Colusa_Study_Area.cpg
└── README.md
```

## Notes

Users should modify the dataset and output paths in each script according to their local environment before running the experiments.

This repository provides model source code, CDL reference data, and vector boundaries defining the spatial extents of the two study areas.

The original Sentinel-1 SAR imagery can be accessed through Google Earth Engine using the provided study area boundaries.

Trained model weights, intermediate training datasets, and experiment outputs are not included in this repository.

## Contact

For questions regarding the code, datasets, or experiments, please contact the authors of the corresponding paper.
