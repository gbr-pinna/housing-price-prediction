# Housing Price Prediction

Machine learning project predicting California housing prices, based on Chapter 2 of *Hands-On Machine Learning with Scikit-Learn and PyTorch* by Aurélien Géron.

This repository contains a Jupyter Notebook that walks through a complete machine learning workflow for regression, from data loading and preprocessing to model evaluation and hyperparameter tuning.

## What’s Inside

- Exploration of the California housing dataset (can be downloaded [here](https://github.com/ageron/data/raw/main/housing.tgz))
- Data cleaning and preprocessing, including handling missing values, scaling numerical features, and encoding categorical features using OneHotEncoder and OrdinalEncoder
- Feature engineering
- Train/test split
- Training and evaluation of multiple models:
  - Linear Regression
  - ElasticNet
  - K-Nearest Neighbors (KNN)
  - Decision Tree Regressor
  - Random Forest Regressor
- Evaluation using cross-validation with RMSE and R²
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Feature importance inspection
- Saving and loading models with joblib

## Requirements

Install the project dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib scipy joblib
