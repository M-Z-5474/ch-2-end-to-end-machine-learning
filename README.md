# ch-2-end-to-end-machine-learning
# 🏡 California Housing Price Prediction with Support Vector Regression (SVR)

This project explores the use of **Support Vector Machine Regression (SVR)** on the California Housing dataset using various pipelines and model optimization techniques. It includes hyperparameter tuning, feature selection, and complete pipeline integration for robust machine learning workflows.

---

## 📁 Dataset

The dataset used is the **California Housing Prices** dataset, which includes features like:

- `longitude`, `latitude`
- `housing_median_age`, `total_rooms`, `total_bedrooms`
- `population`, `households`, `median_income`
- `ocean_proximity` (categorical)
- `median_house_value` (target)

---

## ✅ Project Objectives

1. Train SVR with different kernels (`linear` and `rbf`) and tune hyperparameters (`C`, `gamma`)
2. Compare `GridSearchCV` vs `RandomizedSearchCV`
3. Add a custom transformer to select top features
4. Create a **single unified pipeline** for preprocessing and prediction
5. Automatically optimize preprocessing and modeling steps using `GridSearchCV`

---

## 🔧 Tools & Libraries

- Python 3.x
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib` (optional for visualization)
- Google Colab / Jupyter Notebook

---
🧠 Key Learnings
SVR with RBF kernel captures non-linear patterns better than linear SVR.
RandomizedSearchCV is faster and often nearly as effective as GridSearchCV.
Feature selection improves performance and reduces overfitting.
Pipelines make the entire ML workflow clean and reproducible.


