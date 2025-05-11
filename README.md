# Comparative Study: Regression Analysis for Predicting Ultimate Bearing Capacity of Strip Foundations on Sandy Clay Slopes

![Geotech](https://img.shields.io/badge/Domain-Geotechnical%20Engineering-orange)
![ML](https://img.shields.io/badge/ML-Regression%20Comparison-blue)
![Python](https://img.shields.io/badge/Language-Python-green)
![Models](https://img.shields.io/badge/Algorithms-11%20Models-red)

## Overview

This repository presents a comparative analysis of regression algorithms for predicting the ultimate bearing capacity (Q<sub>ult</sub>) of strip foundations located on sandy clay slopes. The study evaluates multiple machine learning models under standardized conditions to identify the best-performing approach.

## Purpose

The primary aim of this study is to evaluate and compare various regression algorithms to estimate the ultimate bearing capacity of strip foundations on slopes composed of sandy clay. This model helps engineers and practitioners make informed decisions during foundation design in complex geotechnical settings.

## Dataset

The dataset comprises 1,439 samples and includes the following features:

### Input Variables
- **β**: Slope angle (degrees)
- **B**: Foundation width (m)
- **Df**: Foundation depth (m)
- **φ**: Internal friction angle of soil (degrees)
- **γ**: Soil unit weight (kN/m³)
- **P**: Applied vertical load (kN)

### Output Variable
- **Q_ult**: Ultimate bearing capacity (kPa)

## Algorithms Explored

A total of 11 regression algorithms were implemented and evaluated without hyperparameter tuning (default settings):

1. **MLR**: Multiple Linear Regression  
2. **Lasso**: L1-regularized Linear Regression  
3. **Ridge**: L2-regularized Linear Regression  
4. **Polynomial Regression**  
5. **DT**: Decision Tree Regression  
6. **SVR**: Support Vector Regression  
7. **KNN**: K-Nearest Neighbors Regression  
8. **MLP**: Multi-Layer Perceptron  
9. **XGBoost**: Extreme Gradient Boosting  
10. **Random Forest**  
11. **FFNN**: Feed Forward Neural Network  

## Repository Structure

```
/
├── data/
│   └── foundation_slope_dataset.csv
├── model/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Modeling_All_Algorithms.ipynb
│   ├── Evaluation_Results.ipynb
├── LICENSE
└── README.md
```

## Workflow

The analysis pipeline consists of:

1. **EDA & Preprocessing** (`EDA_and_Preprocessing.ipynb`):
   - Data exploration and visualization
   - Anomaly and missing value handling
   - Standardization using `StandardScaler`
   - Data split (80% training / 20% testing)

2. **Model Development** (`Modeling_All_Algorithms.ipynb`):
   - Implementation of all 11 regression models
   - Training using default hyperparameters
   - Prediction and comparison based on uniform input

3. **Model Evaluation** (`Evaluation_Results.ipynb`):
   - Performance metrics (MAE, RMSE, R²)
   - Visualization of predicted vs actual results
   - Summary table of comparative results

## Key Findings

Among all tested models, **XGBoost** achieved the highest accuracy and generalization performance on the test dataset, making it the most reliable choice for this specific geotechnical prediction task.

This study has been published in a peer-reviewed international journal by **Springer Nature**. You can access the full article here:  
👉 [Springer Link](https://link.springer.com/article/10.1007/s40515-025-00544-5)

## Usage

To run the notebooks locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/foundation-slope-regression.git
   cd foundation-slope-regression
   ```

2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
   ```

3. Open Jupyter and run the notebooks:
   ```bash
   jupyter notebook
   ```

## Results

- **Linear models** (MLR, Lasso, Ridge) provided basic performance but were limited in capturing non-linear patterns.
- **Tree-based methods** (DT, RF, XGBoost) significantly outperformed linear models.
- **XGBoost** consistently delivered the best results with minimal overfitting.
- **Neural approaches** (MLP, FFNN) showed good potential but were more sensitive to data preprocessing and scaling.

## Implementation

Although this project does not include a deployed application, the codebase can be extended for integration into design tools or web-based calculators for geotechnical engineers.

## Future Work

- Hyperparameter tuning using GridSearchCV or Bayesian optimization  
- Incorporation of additional soil parameters (e.g., cohesion, saturation)  
- Expansion to different foundation shapes and soil types  
- Real-time deployment for engineering applications

## License

This project is licensed under the terms stated in the [LICENSE](LICENSE) file.

## Acknowledgments

- The authors and collaborators who contributed to the original research publication  
- Springer Nature for publishing the peer-reviewed study  
- The open-source community for providing robust machine learning libraries
