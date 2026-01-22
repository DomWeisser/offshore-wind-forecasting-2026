# Cross-Domain Offshore Wind Power Forecasting: Transfer Learning Through Meteorological Clusters

## Abstract
The global offshore wind sector is expanding rapidly to meet decarbonisation targets, leading to an increasing number of newly commissioned wind farms that require accurate power forecasts from the outset. Reliable early-stage forecasting is essential for grid stability, reserve management, and efficient energy trading. While machine learning methods have demonstrated strong performance in offshore wind power forecasting, they typically rely on large volumes of site-specific data, which are unavailable for new installations. We propose a transfer learning framework based on meteorological clustering to address this data scarcity. Instead of training a single model across all atmospheric conditions, we learn a set of specialised forecasting models, each associated with a distinct weather regime. Pre-trained on large and diverse offshore datasets, these models capture transferable, climate-dependent dynamics and can be adapted efficiently to new sites. By leveraging their built-in calibration to seasonal and meteorological variability, our approach removes the industry-standard requirement of a full year of local measurements. We evaluate the framework across eight offshore wind farms, demonstrating that accurate cross-domain forecasting can be achieved using just over seven months of site-specific data. The proposed method attains an average mean absolute error of 3.83%, confirming that reliable power forecasts are possible without observing a complete annual cycle. Beyond power forecasting, this meteorology-aware transfer learning approach now opens new opportunities for offshore wind applications such as early-stage wind resource assessment, where reducing data requirements can significantly accenlerate and de-risk project development.

## Data Availability




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

This notebook trains the library of GP models using two years of historical data. It performs a grid search to determine the optimal time-period length (p) and number of latent clusters (K) by maximising the composite quality score defined in the methodology. Using this optimal configuration, it trains a VAE to extract latent features and subsequently trains distinct GP models for each identified weather cluster.

- 03: Transfer Learning

This notebook implements the transfer learning methodology across the eight target wind farms (Horns Rev, Seagreen, Gemini, Hollandse Noord, Dieppe, Moray Firth, Kriegers Flak and East Anglia One). Additionally, it executes a comparative baseline by training separate GP models from scratch on the target data to benchmark the transfer performance.


### Results 
A breakdown of the results for all three random seeds (42, 63, 84) used in the final paper can be found in "Results of transfer learning.xslx". This file contains the source GP results as well as the transfer learning forecasting performance for all eight target farms, stratified by cluster and data capacity.


## License

