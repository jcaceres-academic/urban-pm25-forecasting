# PM2.5 Pollution Analysis and Prediction in Madrid

Open, reproducible research materials supporting the article:

**Analysis and Prediction of PM2.5 Pollution in Madrid: Use of Prophet–LSTM Hybrid Models** *(AppliedMath, 2024)*

## 🔗 Project links

```         
🌐 Project page (methodology, results and context)
https://jcaceres-academic.github.io/urban-pm25-forecasting/
📘 Reproducible notebook
https://jcaceres-academic.github.io/urban-pm25-forecasting/notebooks.html
📄 Published article
https://doi.org/10.3390/appliedmath4040076
```

## 📁 Repository overview

-   docs/ – Rendered Quarto website (GitHub Pages deployment)
-   data/ – Processed datasets (PM2.5 and meteorological variables)
-   scripts/ – Reproducible data processing and modelling pipeline
-   figures/ – Publication-ready figures (PNG / TIFF)

## 🧠 Research scope

This repository supports a study on **urban air quality analysis and prediction in Madrid**, focusing on:

-   Temporal analysis of PM2.5 levels (2019–2024)
-   Spatial distribution across monitoring districts
-   Hybrid modelling using *Prophet–LSTM*
-   Integration of meteorological and environmental data

The approach combines *statistical forecasting and deep learning* to capture both:

-   Seasonal and trend components (Prophet)
-   Long-term dependencies and nonlinear patterns (LSTM)

## 🔁 Reproducibility principles

This project follows principles of:

-   Reproducibility by design
-   Transparent data processing
-   Programmatic figure generation
-   Open data integration (Madrid Open Data Portal)

All results presented in the article are generated from the scripts and notebooks included in this repository.

### Data and reproducibility

All datasets and scripts are available in the associated Zenodo repository:

https://doi.org/10.5281/zenodo.19659982

Note: The Zenodo archive contains additional scripts from related research lines.  
The scripts relevant to this study are those included in this repository and described in the documentation.

## ⚙️ Technologies

-   Python 3.12
-   pandas · numpy
-   Prophet
-   TensorFlow / Keras (LSTM)
-   matplotlib · seaborn
-   Quarto (documentation and web deployment)

## 📜 License

Creative Commons Attribution 4.0 (CC BY 4.0)

------------------------------------------------------------------------

➡️ For the full project description, workflow, references, and educational context,\
see the **project website** linked above.
