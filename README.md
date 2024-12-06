# Project Proposal: Identifying Optimal Airbnb Investment Opportunities in New York City Using Airbnb Open Data

## Midterm Update:
https://youtu.be/g-1oD2hXkjE


## Description of the Project

This project aims to help potential Airbnb hosts identify the best locations and property types in New York City (NYC) to maximize their income. By analyzing the Airbnb Open Data—which includes detailed information about listings, hosts, and pricing—we will develop models to predict predict rental costs based on listing features. This analysis will provide data-driven recommendations for investors looking to enter the Airbnb market in NYC.

## Clear Goals

### Primary Goal

- **Predict rental income based on listing features in NYC.**

### Secondary Goals

- Predict potential rental income and occupancy rates based on listing features.
- Identify key factors that influence the success of Airbnb listings, such as location, room type, and host characteristics.

## Data Collection

### What Data Needs to Be Collected

We will use the **Airbnb Open Data** available on Kaggle:

- **Listings Dataset:**
  - **Columns:**
    - `id`, `NAME`, `host id`, `host_identity_verified`, `host name`, `neighbourhood group`, `neighbourhood`, `lat`, `long`, `country`, `country code`, `instant_bookable`, `cancellation_policy`, `room type`, `Construction year`, `price`, `service fee`, `minimum nights`, `number of reviews`, `last review`, `reviews per month`, `review rate number`, `calculated host listings count`, `availability 365`, `house_rules`, `license`

### How the Data Will Be Collected

- **Download the Dataset:**
  - Access and download the Airbnb Open Data from Kaggle: [Airbnb Open Data](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata).

## Modeling Plan

### How We Plan on Modeling the Data

#### Data Preprocessing

- **Cleaning:**
  - Handle missing values in key columns such as `price`, `availability 365`, and `reviews per month`.
  - Correct data types for columns like `price` and `Construction year`.

- **Feature Engineering:**
  - **Calculate Potential Income:**
    - Estimate annual rental income:
      ```
      Estimated Income = (price + service fee) * availability_365 / minimum_nights
      ```
  - **Occupancy Rate:**
    - Approximate occupancy rate using `availability_365` and `minimum_nights`.
  - **Create Categorical Variables:**
    - Encode categorical variables like `neighbourhood group`, `room type`, and `cancellation_policy` using one-hot encoding or label encoding.
  - **Host Experience:**
    - Use `calculated host listings count` to gauge host experience.

#### Exploratory Data Analysis (EDA)

- Analyze distribution of prices, availability, and number of reviews across different neighborhoods and property types.
- Identify correlations between listing features and estimated income.

#### Predictive Modeling

- **Regression Models:**
  - **Linear Regression:** To predict potential rental income based on listing features.
  - **Random Forest Regressor:** To capture non-linear relationships and feature interactions.
  - **Gradient Boosting Machines (e.g., XGBoost):** For improved predictive accuracy.

- **Classification Models (Optional):**
  - **Decision Trees** or **Logistic Regression:** To classify listings into high or low ROI categories.

- **Model Selection:**
  - Compare models using cross-validation and select the best-performing one based on evaluation metrics.

#### Recommendation System

- Develop a system that suggests optimal neighborhoods and property types based on predicted rental income and investor budget.

### Tools and Techniques

- **Programming Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, Folium (for mapping)

## Data Visualization

### How We Plan on Visualizing the Data

- **Geospatial Maps:**
  - Use Folium to create interactive maps showing:
    - Distribution of listings across NYC.
    - Estimated rental income by location.

- **Heatmaps:**
  - Visualize concentrations of high and low estimated income areas.

- **Bar Charts and Box Plots:**
  - Compare prices and availability across `neighbourhood groups` and `room types`.

- **Scatter Plots:**
  - Plot relationships between `price`, `number of reviews`, `availability 365`, and estimated income.

- **Correlation Matrix:**
  - Use a heatmap to identify relationships between variables.

- **Histograms:**
  - Display distributions of key numerical features like `price`, `minimum nights`, and `reviews per month`.

## PCA Analysis

### Purpose of PCA

Principal Component Analysis (PCA) is utilized in this project to reduce the dimensionality of the dataset while preserving as much variance as possible. By transforming the dataset into a set of uncorrelated variables (principal components), we can more easily visualize and interpret the relationships between different features.

### PCA Implementation Steps

1. **Data Preparation**:
   - The dataset was cleaned and preprocessed, including handling missing values and converting categorical variables into numerical format using one-hot encoding.

2. **Feature Selection**:
   - Relevant numerical features included latitude, longitude, price, service fee, minimum nights, number of reviews, reviews per month, review rate number, calculated host listings count, and availability.

3. **PCA Execution**:
   - PCA was performed using the `PCA` class from the `sklearn.decomposition` module. A pipeline was created that included preprocessing steps followed by PCA application.

4. **Results**:
   - The first two principal components were plotted to visualize the distribution of the Airbnb listings.
   - The explained variance for each component was calculated, revealing how much variance is captured by the first and second components.

## Test Plan

### What Is the Test Plan?

#### Data Splitting

- **Training Set:** 80% of the data for training the models.
- **Testing Set:** 20% of the data for evaluating model performance.

#### Validation Strategy

- **Cross-Validation:**
  - Implement k-fold cross-validation (e.g., k=5) to assess model stability and prevent overfitting.

#### Evaluation Metrics

- **For Regression Models:**
  - **Mean Absolute Error (MAE):** Measures average magnitude of errors.
  - **Root Mean Squared Error (RMSE):** Penalizes larger errors more than MAE.
  - **R-squared (R²):** Indicates proportion of variance explained by the model.

#### Model Tuning

- Use grid search or randomized search to optimize hyperparameters for algorithms like Random Forest and XGBoost.

#### Model Validation

- **Residual Analysis:**
  - Plot residuals to check for patterns that might indicate model bias.

- **Feature Importance:**
  - Identify which features contribute most to predicting rental income.

- **Outlier Detection:**
  - Analyze and possibly remove outliers that could skew model results.

---


project_root/
├─ data/
│  ├─ raw/               # Store original downloaded data files here
│  ├─ processed/         # Store cleaned and feature-engineered datasets here
├─ src/
│  ├─ data_collection/   # Scripts related to downloading or accessing Airbnb data
│  ├─ data_cleaning/     # Scripts for cleaning and preprocessing the raw data
│  ├─ feature_engineering/ # Scripts for creating new features, transformations
│  ├─ visualization/     # Scripts/notebooks for creating plots and interactive visuals
│  ├─ modeling/          # Scripts for training, evaluating, and saving models
│  ├─ utils/             # Helper functions that can be shared across modules
├─ tests/
│  ├─ test_data_cleaning/   # Test scripts ensuring cleaning steps are correct
│  ├─ test_feature_engineering/ # Tests for validating feature transformations
│  ├─ test_modeling/         # Tests ensuring model trains and predicts correctly
│  ├─ test_visualization/    # Optional tests for checking if visualization scripts run without error
├─ notebooks/
│  ├─ eda.ipynb          # Exploratory data analysis notebook
│  ├─ modeling_experiments.ipynb # Experimentation with different models
│  ├─ visualization_experiments.ipynb # Experimentation with plots/interactive visuals
├─ environment.yml        # Conda environment file or requirements.txt for Python packages
├─ Makefile               # Make commands to set up environment, run pipeline, run tests
├─ README.md              # Project overview, setup instructions, methodology, results, contribution guidelines
├─ .github/
│  └─ workflows/
│     └─ test.yml         # GitHub Actions workflow to run tests automatically
└─ figures/
   ├─ eda_plots/          # Static figures generated by EDA scripts or notebooks
   ├─ model_evaluation/   # Plots related to model performance

