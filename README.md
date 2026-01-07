# Cross-Domain Offshore Wind Power Forecasting: Transfer Learning Through Meteorological Clusters

## Abstract
The global offshore wind sector is expanding rapidly to meet decarbonisation targets, leading to an increasing number of newly commissioned wind farms that require accurate power forecasts from the outset. Reliable early-stage forecasting is essential for grid stability, reserve management, and efficient energy trading. While machine learning methods have demonstrated strong performance in offshore wind power forecasting, they typically rely on large volumes of site-specific data, which are unavailable for new installations. We propose a transfer learning framework based on meteorological clustering to address this data scarcity. Instead of training a single model across all atmospheric conditions, we learn a set of specialised forecasting models, each associated with a distinct weather regime. Pre-trained on large and diverse offshore datasets, these models capture transferable, climate-dependent dynamics and can be adapted efficiently to new sites. By leveraging their built-in calibration to seasonal and meteorological variability, our approach removes the industry-standard requirement of a full year of local measurements. We evaluate the framework across eight offshore wind farms, demonstrating that accurate cross-domain forecasting can be achieved using just over seven months of site-specific data. The proposed method attains an average mean absolute error of 3.83%, confirming that reliable power forecasts are possible without observing a complete annual cycle. Beyond power forecasting, this meteorology-aware transfer learning approach now opens new opportunities for offshore wind applications such as early-stage wind resource assessment, where reducing data requirements can significantly accenlerate and de-risk project development.

## Data Availability


## Installation
To reproduce this work, first clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage



## License

