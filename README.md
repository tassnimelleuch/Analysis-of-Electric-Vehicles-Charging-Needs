# 🔌 EV Charging Infrastructure Demand Predictor

![Demo Visualization](https://via.placeholder.com/800x400.png?text=EV+Charging+Cluster+Map)  
*Predicting cities needing more charging stations using machine learning*

## 📌 Overview
This project analyzes **Electric Vehicle (EV) adoption patterns** across US cities to:
1. Cluster cities based on EV characteristics using K-Means
2. Predict which cities need more charging infrastructure using:
   - K-Nearest Neighbors (KNN)
   - Support Vector Machines (SVM)
   - XGBoost
3. Visualize high-priority locations on an interactive map

## 🗃️ Dataset
**Source:** [EV Population Data - Kaggle](https://www.kaggle.com/datasets/yashdogra/ev-bhebic-c/data)  
**Original Features:**
- City/State locations
- Vehicle model years
- Electric ranges
- Utility providers
- EV types (BEV/PHEV)

## 🛠️ Preprocessing
1. **Cleaning:**
   - Handled missing values (mode imputation for categorical, median for numerical)
   - Dropped invalid geographical coordinates
2. **Feature Engineering:**
   - Extracted latitude/longitude from `Vehicle Location`
   - Created aggregated city-level metrics:
     - Average electric range
     - Percentage of BEVs
     - Number of unique utilities
3. **Transformations:**
   - Log-transformed skewed features
   - StandardScaler for normalization
   - PCA for clustering visualization (n_components=2)

## 🏷️ Label Creation
Cities are flagged as needing more charging stations if they meet **ANY** of:
- Below-median electric range (`< 44.8 miles`)
- Above-median BEV percentage (`> 79.1%`)
- Below-median utility providers (`< 1`)
- Older vehicle fleet (model year `< 2021`)

## 🤖 Models Used
| Model       | Accuracy | AUC-ROC | Best For |
|-------------|----------|---------|----------|
| **K-Means** | -        | -       | City clustering |
| **KNN**     | 0.92     | 0.94    | Baseline |
| **SVM**     | 0.89     | 0.91    | High-dim data |
| **XGBoost** | 0.95     | 0.98    | Final predictions |

## 📊 Visualizations
1. **Exploratory Analysis:**
   - Feature distribution plots
   - Correlation heatmap
2. **Clustering:**
   - Elbow method for optimal K
   - Silhouette analysis (score: 0.45)
   - 2D PCA cluster visualization
3. **Predictions:**
   - Confusion matrices
   - ROC curves
   - Precision-recall curves
4. **Geospatial:**
   - Interactive Folium map with:
     - Marker clusters of high-need cities
     - Choropleth of state-level demand

## 🚀 Installation
```bash
git clone https://github.com/yourusername/ev-charging-predictor.git
cd ev-charging-predictor
pip install -r requirements.txt
