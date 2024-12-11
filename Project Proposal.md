# Pricing-Focused Airbnb Investment Opportunities in NYC

## Description

This project aims to identify the most lucrative investment opportunities for Airbnb hosts in New York City by focusing on pricing strategies. Using **Airbnb Open Data**, we develop models to predict optimal rental prices based on listing features and market dynamics. The goal is to provide actionable insights and pricing recommendations tailored to different neighborhoods and property types, enabling hosts to optimize profitability.

---

## Goals

### Primary Goal
- **Develop predictive models for optimal pricing of Airbnb listings in NYC.**

### Secondary Goals
- Analyze price distributions across neighborhoods and property types.
- Identify key factors influencing listing prices.
- Provide pricing recommendations to maximize profitability.

---

## Data Collection

### What Data Needs to Be Collected
We will use the publicly available **Airbnb Open Data** from Kaggle:
- [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

### Key Columns for Analysis
- `price`
- `neighbourhood group`
- `neighbourhood`
- `room type`
- `minimum nights`
- `availability_365`
- `reviews_per_month`
- `calculated_host_listings_count`

### Data Collection Method
- **Raw Data**: Download from Kaggle and place in the `data/raw` folder.
- **Processed Data**: Transform raw data through preprocessing and feature engineering steps to produce clean datasets for analysis.

---

## Modeling Plan

### Data Preprocessing
- **Cleaning**:
  - Handle missing values in critical columns like `price` and `reviews_per_month`.
  - Convert `price` to numerical format for analysis.
- **Feature Engineering**:
  - Calculate normalized pricing metrics (e.g., price per night).
  - Aggregate neighborhood-level metrics for comparison.
  - Encode categorical variables such as `neighbourhood group` and `room type`.

### Modeling Techniques
We will utilize the following models:
1. **Linear Regression**: For straightforward price prediction.
2. **Random Forest Regressor**: To capture non-linear relationships.
3. **Gradient Boosting Machines (e.g., XGBoost)**: For high-accuracy predictions.

#### Key Outputs
- Predicted optimal nightly prices based on listing features.
- Analysis of price sensitivity by neighborhood and property type.

---

## Visualization Plan

We plan to visualize the data using:
1. **Geospatial Heatmaps**: Displaying price distributions across neighborhoods.
2. **Box Plots**: Comparing pricing across room types and neighborhoods.
3. **Scatter Plots**: Exploring relationships between features (e.g., reviews and price).
4. **Histograms**: Analyzing price distributions.
5. **Interactive Visualizations**: Built into a pricing dashboard.

---

## Test Plan

We will evaluate the models using:
1. **Data Splitting**:
   - Train the models on 80% of the dataset.
   - Test the models on the remaining 20%.
2. **Validation**:
   - Implement k-fold cross-validation (k=5) to ensure model stability.
3. **Evaluation Metrics**:
   - **Mean Absolute Error (MAE)**: Measures average prediction error.
   - **Root Mean Squared Error (RMSE)**: Penalizes larger errors more.
   - **R-squared (R²)**: Indicates how well the model explains variability in the data.
4. **Residual Analysis**:
   - Evaluate residual patterns to ensure unbiased predictions.

---

## Deliverables

1. **Pricing Dashboard**:
   - An interactive tool showing recommended prices by location and property type.
2. **Comprehensive Report**:
   - Detailed insights into pricing trends and model predictions.
3. **Codebase**:
   - Modularized Python scripts for data analysis and modeling.

---

## Tools and Techniques

- **Programming Language**: Python
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib, seaborn, Folium (for mapping)
- **Modeling Frameworks**: scikit-learn, XGBoost

---
