# Pricing-Focused Airbnb Investment Opportunities in NYC

## Description

This project aims to identify the most lucrative investment opportunities for Airbnb hosts in New York City by focusing on pricing strategies. Using **Airbnb Open Data**, we develop models to predict optimal rental prices based on listing features and market dynamics. Our findings will provide actionable insights and pricing recommendations tailored to different neighborhoods and property types.

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

### Data Source
- **Airbnb Open Data**: Accessible via Kaggle at [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), which includes comprehensive details about listings, hosts, and pricing metrics.

### Key Columns for Analysis
- `price`
- `neighbourhood group`
- `neighbourhood`
- `room type`
- `minimum nights`
- `availability_365`
- `reviews_per_month`
- `calculated_host_listings_count`

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

### Predictive Models
#### Regression Models
- **Linear Regression**: For straightforward price prediction.
- **Random Forest Regressor**: To capture non-linear relationships.
- **Gradient Boosting (XGBoost)**: For high-accuracy predictions.

#### Key Outputs
- Optimal nightly price predictions based on listing features.
- Price sensitivity analysis by neighborhood and room type.

---

## Visualization

- **Geospatial Heatmaps**: Visualizing price distributions by neighborhood.
- **Box Plots**: Comparing pricing across room types and neighborhoods.
- **Scatter Plots**: Exploring relationships between variables (e.g., reviews and price).
- **Histograms**: Analyzing price distributions.

---

## Deliverables

1. **Pricing Dashboard**:
   - Interactive tool showing recommended prices by location and property type.
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

## Test Plan

- **Data Splitting**: 80% training, 20% testing.
- **Validation**: k-fold cross-validation (k=5).
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- **Residual Analysis**:
  - Validate model accuracy and detect biases.
