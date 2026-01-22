# Cross-Domain Offshore Wind Power Forecasting: Transfer Learning Through Meteorological Clusters

## Abstract
Ambitious decarbonisation targets are catalysing growth in orders of new offshore wind farms. For these newly commissioned plants to run, accurate power forecasts are needed from the onset. These allow grid stability, good reserve management and efficient energy trading. Despite machine learning models having strong performances, they tend to require large volumes of site-specific data that new farms do not yet have. To overcome this data scarcity, we propose a novel transfer learning framework that clusters power output according to covariate meteorological features. Rather than training a single, general-purpose model, we thus forecast with an ensemble of expert models, each trained on a cluster. As these pre-trained models each specialise in a distinct weather pattern, they adapt efficiently to new sites and capture transferable, climate-dependent dynamics. Through the expert models’ built-in calibration to seasonal and meteorological variability, we remove the industry-standard requirement of local measurements over a year. Our contributions are two-fold — we propose this novel framework and comprehensively evaluate it on eight offshore wind farms, achieving accurate cross-domain forecasting with under five months of site-specific data. Our experiments achieve a MAE of 3.52\%, providing empirical verification that reliable forecasts do not require a full annual cycle. Beyond power forecasting, this climate-aware transfer learning method opens new opportunities for offshore wind applications such as early-stage wind resource assessment, where reducing data requirements can significantly accelerate project development whilst effectively mitigating its inherent risks.

## Data Availability
The dataset utilised in this analysis is derived from the study "Analyzing Europe's Biggest Offshore Wind Farms: a Data set With 40 Years of Hourly Wind Speeds and Electricity Production". The full dataset is publically available via Figshare:

- https://doi.org/10.6084/m9.figshare.19139648

This project specifically utilises the individual wind farm files located in the "Data_Per_Wind_Farm" directory.

To facilitate the reproduction of results, the data required for this analysis (notebooks 2 and 3) have been included in this repository. These files contains hourly records for the selected source and target wind farms, filtered to the last 5 years of available data (01.01.2015 - 31.12.2019).

## Usage
To reproduce the results presents in the paper, please follow the steps below
### Installation
First clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Scripts
The project is divided into three stages, each contained within a specific Jupyter Notebook.

- 01: Source Farm Selection

This notebook executes a clustering analysis on the 29 candidate wind farms to identify distinct meteorological and operational similarities. It partitions the farms into six clusters, with one representative source farm from each selected: Beatrice, Baltic Eagle, Hornsea (Project 1), Gode, Walney and Iles et de Noirmoutier.

- 02: Training Source Models

This notebook trains the library of GP models using two years of historical data. It performs a grid search to determine the optimal time-period length (p) and number of latent clusters (K) by maximising the composite quality score defined in the methodology. Using this optimal configuration, it trains a VAE to extract latent features and subsequently trains distinct GP models for each identified weather cluster. To facilitate reproducibility, the pre-trained VAE models for all three random seeds have been archived in the model directory.

- 03: Transfer Learning

This notebook implements the transfer learning methodology across the eight target wind farms (Horns Rev, Seagreen, Gemini, Hollandse Noord, Dieppe, Moray Firth, Kriegers Flak and East Anglia One). Additionally, it executes a comparative baseline by training separate GP models from scratch on the target data to benchmark the transfer performance. To facilitate reproducibility, the requisite metadata and pre-trained VAE artificats needed in the transfer pipeline for all three seeds are available in the models directory. 

### Results 
A breakdown of the results for all three random seeds (42, 63, 84) used in the final paper can be found in "Results of transfer learning.xslx". This file contains the source GP results as well as the transfer learning forecasting performance for all eight target farms, stratified by cluster and data capacity.
