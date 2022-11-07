# Master thesis repository 
### The usage of Machine learning techniques in analysis of transcriptomic profiles of tumor cells

### This repository contains:
- Script analysis and Machine Learning Models
- Results
- Plots

### Repository map:
- [Information about data](./data)
- [PCA + UMAP scatter plots](./plots/pca_umap)
- [Classification confusion Matrix]()
- [Other plots](./plots/age_plots)
- [Results](./classification/results)
- [Classification and additional algorithms](./classification/classif)
  - [Classification factory](./classification/classif/classif_factory.py) - script that runs each case
  - [Balancing classes](./classification/classif/sampling.py)
  - [Feature Selection](./classification/classif/feature_selection.py)
  - [Data Transformation](./classification/classif/normalization.py)
  - [Dimensionality reduction](./classification/classif/dim_red.py)
  - [Classification algorithms](./classification/classif/classif.py)

- [Additional Classification Jupiter Notebook](./classification/classfication_checkpoints/1_Classification.ipynb)

---

### Analysis overview:
1. Data collection.
2. Data visualization.
3. Data preprocessing using various algorithms and tuning them.
4. Classification using various algorithms and tuning them.
5. Result verification.

Most of the analysis were done using Jupiter Notebook. In the process of tuning algorithm variables a lot of work that was done
isn't saved, because variables were changing very often.

Additionally, I have created script, that produces result for every possible case of data preprocessing and classification.
All the results are in [Results](./classification/results) folder.
