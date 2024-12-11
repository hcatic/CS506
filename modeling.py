#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go

# Importing machine learning models
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn import metrics

# Importing metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Importing model selection utilities
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve


# Define adjusted R^2
def adjusted_r2(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


df = pd.read_csv('data/processed/preprocessed_data.csv') # Add ../ prefix if running notebook directly

# log-transform highly skewed variables
df['log_price'] = np.log1p(df['price'])
df['log_host_listings'] = np.log1p(df['calculated_host_listings_count'])
df['log_avail365'] = np.log1p(df['availability_365'])

y = df['log_price']

# Drop y-var and any other variables derived from price
X = df.drop(["log_price", "price", "neighbourhood", "total_cost_min_stay", "annual_potential_income", "min_potential_income", "availability_365", "calculated_host_listings_count"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('RT', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("CatBoost", CatBoostRegressor(verbose=False))]

rmse_scores = []
r2_scores = []
mae_scores = []
mse_scores = []
execution_times = []

"""for name, regressor in models:
    start_time = time.time()

    # Fit the model
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Calculate RMSE
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    rmse_scores.append(rmse)
    
    # Calculate R^2 score
    r2 = metrics.r2_score(y_test, y_pred)
    r2_scores.append(r2)

    # Calculate MAE
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    # Calculate MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    # Calculate the execution time of the model
    execution_time = time.time() - start_time
    execution_times.append(execution_time)

    print(f"RMSE: {round(rmse, 4)} ({name})")
    print(f"R^2 Score: {round(r2, 4)} ({name})")
    print(f"MAE: {round(mae, 4)} ({name})")
    print(f"MSE: {round(mse, 4)} ({name})")
    print(f"Execution Time: {round(execution_time, 2)} seconds\n")"""


def plot_learning_curves(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), 
                        scoring='neg_mean_squared_error', figsize=(12, 6)):
    """
    Plot learning curves showing train and validation scores vs training size.
    
    Parameters:
    -----------
    estimator : estimator object
        A scikit-learn estimator
    X : array-like
        Training data
    y : array-like
        Target values
    cv : int
        Number of cross-validation folds
    train_sizes : array-like
        Points at which to evaluate training size
    scoring : string
        Scoring metric to use
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate learning curves
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=train_sizes,
        scoring=scoring,
        shuffle=True
    )
    
    # Calculate mean and std for training scores
    train_mean = np.mean(-train_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    
    # Calculate mean and std for validation scores
    valid_mean = np.mean(-valid_scores, axis=1)
    valid_std = np.std(-valid_scores, axis=1)
    
    # Plot learning curves with confidence bands
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, valid_mean, label='Validation Score', color='red')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=20, figsize=(10, 6)):
    """
    Plot feature importance for models that support it.
    
    Parameters:
    -----------
    model : fitted model object
        Model with feature_importances_ or coef_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1][:top_n]
    
    # Plot feature importance
    plt.bar(range(top_n), importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top Feature Importance')
    plt.tight_layout()
    
    return plt.gcf()

def plot_residuals(y_true, y_pred, figsize=(15, 5)):
    """
    Create three plots for residual analysis:
    1. Residuals vs Predicted
    2. Residual Distribution
    3. Q-Q Plot
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple
        Figure size
    """
    residuals = y_true - y_pred
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3)
    
    # Residuals vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # Residual Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel('Residuals')
    ax2.set_title('Residual Distribution')
    
    # Q-Q Plot
    ax3 = fig.add_subplot(gs[0, 2])
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    plt.tight_layout()
    return plt.gcf()

def plot_prediction_error(y_true, y_pred, figsize=(10, 6)):
    """
    Create an interactive scatter plot of predicted vs actual values
    with error analysis using Plotly.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple
        Figure size
    """
    fig = go.Figure()
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[min(y_true), max(y_true)],
        y=[min(y_true), max(y_true)],
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    # Actual vs Predicted scatter
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color=np.abs(y_true - y_pred),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='|Error|')
        )
    ))
    
    fig.update_layout(
        title='Prediction Error Analysis',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        width=figsize[0]*100,
        height=figsize[1]*100,
        showlegend=True
    )
    
    return fig

def create_model_dashboard(results, X, y, feature_names):
    """
    Create a comprehensive dashboard for model comparison and analysis.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing model results
    X : array-like
        Feature matrix
    y : array-like
        Target values
    feature_names : list
        List of feature names
    """
    for result in results:
        model = result['model']
        name = result['name']
        y_pred = model.predict(X)
        
        print(f"\n=== {name} Analysis ===")
        
        # Learning curves
        plt.figure(figsize=(12, 6))
        plot_learning_curves(model, X, y)
        plt.title(f'{name} Learning Curves')
        plt.show()
        
        # Feature importance (if applicable)
        try:
            plt.figure(figsize=(10, 6))
            plot_feature_importance(model, feature_names)
            plt.title(f'{name} Feature Importance')
            plt.show()
        except:
            print(f"Feature importance not available for {name}")
        
        # Residual analysis
        plot_residuals(y, y_pred)
        plt.suptitle(f'{name} Residual Analysis')
        plt.show()
        
        # Prediction error analysis
        fig = plot_prediction_error(y, y_pred)
        fig.update_layout(title=f'{name} Prediction Error Analysis')
        fig.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_metrics_comparison(models, rmse_scores, r2_scores, mae_scores, execution_times):
    """
    Create a summary plot comparing key metrics across models
    """
    model_names = [model[0] for model in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE Plot
    axes[0, 0].bar(model_names, rmse_scores)
    axes[0, 0].set_title('RMSE by Model')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # R² Plot
    axes[0, 1].bar(model_names, r2_scores)
    axes[0, 1].set_title('R² Score by Model')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # MAE Plot
    axes[1, 0].bar(model_names, mae_scores)
    axes[1, 0].set_title('MAE by Model')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Time Plot
    axes[1, 1].bar(model_names, execution_times)
    axes[1, 1].set_title('Execution Time (s)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_models(models, X_train, X_test, y_train, y_test, feature_names, 
                  rmse_scores, r2_scores, mae_scores, execution_times):
    """
    Create comprehensive analysis using the provided plotting functions
    """
    # First plot the overall metrics comparison
    plot_metrics_comparison(models, rmse_scores, r2_scores, mae_scores, execution_times)
    
    # Analyze each model individually
    for (name, model), rmse, r2, mae, exec_time in zip(models, rmse_scores, r2_scores, 
                                                      mae_scores, execution_times):
        print(f"\n=== {name} Analysis ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Execution Time: {exec_time:.2f}s")
        
        # Plot learning curves using the provided function
        print("\nLearning Curves:")
        plot_learning_curves(model, X_train, y_train)
        plt.show()
        
        # Try to plot feature importance if the model supports it
        #print("\nFeature Importance:")
        #try:
        #    plot_feature_importance(model, feature_names)
        #    plt.show()
        #except:
        #    print(f"Feature importance not available for {name}")
        
        # Make predictions and plot residuals
        y_pred = model.predict(X_test)
        print("\nResidual Analysis:")
        plot_residuals(y_test, y_pred)
        plt.show()
        
        # Plot prediction error
        print("\nPrediction Error Analysis:")
        plot_prediction_error(y_test, y_pred)
        plt.show()
        
        print("\n" + "="*50 + "\n")


#analyze_models(models, X_train, X_test, y_train, y_test, X.columns,
              #rmse_scores, r2_scores, mae_scores, execution_times)


# Initialize the models
# Models that performed poorly in previous steps were removed 
# (Random Forrest takes significantly longer as other similarly performant models)

models = [#('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Initialize lists to store metrics
rmse_scores = []
r2_scores = []
r2_adj_scores = []
mae_scores = []
mse_scores = []
execution_times = []

trained_models = []

# Define the hyperparameters for each model
param_grids = {
    #'GBM': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.15, 0.2]},
    'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.15, 0.2]},
    'CatBoost': {'iterations': [100, 200, 300], 'learning_rate': [0.1, 0.15, 0.2], 'depth': [4, 6]}
}

# Train and evaluate the models with hyperparameter tuning
for name, regressor in models:
    print(f"Hyperparameter Tuning for {name}:")
    start_time = time.time()

    if param_grids[name]:
        grid_search = GridSearchCV(regressor, param_grid=param_grids[name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_model = regressor.fit(X_train, y_train)

    trained_models.append(best_model)  # Store the trained model

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    # Calculate R^2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    # Calculate R^2 Adj score
    r2_adj = adjusted_r2(y_test, y_pred, len(X_train), len(X_train.columns))
    r2_adj_scores.append(r2_adj)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    # Calculate the execution time of the model
    execution_time = time.time() - start_time
    execution_times.append(execution_time)

    print(f"RMSE: {round(rmse, 4)} ({name})")
    print(f"R^2 Score: {round(r2, 4)} ({name})")
    print(f"Adjusted R^2: {round(r2_adj, 4)} ({name})")
    print(f"MAE: {round(mae, 4)} ({name})")
    print(f"MSE: {round(mse, 4)} ({name})")
    print(f"Execution Time: {round(execution_time, 2)} seconds\n")


# Final Prediction Model
best_model = trained_models[np.argmin(rmse_scores)]

# Make predictions on the test set using the final model
y_best_pred = best_model.predict(X_test)
final_y_pred = y_best_pred
final_y_test = y_test

# Create a DataFrame with the predicted prices and true prices
results = pd.DataFrame({'Predicted Price': final_y_pred, 'True Price': final_y_test})

# Calculate the difference between the true prices and predicted prices and add a new column
results['Difference'] = results['True Price'] - results['Predicted Price']

# Display the results
results.head()


def plot_importance(model, features, num=50, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import os

# Ensure the /results directory exists
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Reverse transformation for the log-transformed target variable
def reverse_log_transform(value):
    return np.expm1(value)  # exp(value) - 1

# Function to check if a feature is categorical based on range
def is_categorical_feature(feature_data):
    return feature_data.min() >= 0 and feature_data.max() <= 1

# Function to generate and save box-and-whisker plots for categorical variables
def plot_box_whisker_for_categorical(X, y, feature, results_dir):
    """
    Plot and save box-and-whisker plot for a categorical feature.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataset (Test or Train set).
    y : pd.Series or np.array
        Target variable.
    feature : str
        Name of the categorical feature.
    results_dir : str
        Directory to save the plots.
    """
    data = pd.DataFrame({feature: X[feature], 'price': y})
    data['price'] = data['price'].apply(reverse_log_transform)  # Reverse transform for better interpretability

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=feature, y='price', data=data)
    plt.title(f"Distribution of Price by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, f"{feature}_box_whisker_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Box-and-whisker plot for {feature} saved to {plot_path}")

# Update main function to call the box-and-whisker plot for categorical variables
def plot_feature_impact_with_box_whisker(model, X, y, top_n=5):
    """
    Generate and save box-and-whisker plots for categorical features
    and partial dependence plots for continuous features.

    Parameters:
    -----------
    model : Trained model
        The fitted model with feature importance or partial dependence capability.
    X : pd.DataFrame
        Feature dataset (Test or Train set).
    y : pd.Series or np.array
        Target variable.
    top_n : int
        Number of top features to analyze.
    """
    if hasattr(model, 'feature_importances_'):
        # Identify top features by importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        top_features = feature_importances.head(top_n)['Feature'].values
    else:
        raise ValueError("The model does not support feature importances.")

    for feature in top_features:
        feature_data = X[feature]
        is_categorical = is_categorical_feature(feature_data)

        if is_categorical:
            # Plot and save box-and-whisker plot for categorical feature
            plot_box_whisker_for_categorical(X, y, feature, results_dir)
        else:
            # Plot partial dependence for continuous feature
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                features=[feature],
                grid_resolution=50,
                ax=ax
            )
            display.axes_[0, 0].set_ylabel("Price (Reverse Log Scale)")
            yticks = display.axes_[0, 0].get_yticks()
            display.axes_[0, 0].set_yticklabels([round(reverse_log_transform(y), 2) for y in yticks])
            
            # Save plot
            plot_path = os.path.join(results_dir, f"{feature}_continuous_partial_dependence.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Partial dependence plot for {feature} saved to {plot_path}")

# Function to plot and save the top 10 most important features
def plot_top_10_features(model, X, results_dir):
    """
    Generate and save a bar plot of the top 10 most important features.

    Parameters:
    -----------
    model : Trained model
        The fitted model with feature_importances_.
    X : pd.DataFrame
        Feature dataset.
    results_dir : str
        Directory to save the plot.
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Select top 10 features
        top_10_features = feature_importances.head(10)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_10_features,
            x='Importance',
            y='Feature',
            palette='viridis'
        )
        plt.title("Top 10 Most Important Features")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(results_dir, "top_10_features_importance.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Top 10 feature importance plot saved to {plot_path}")
    else:
        raise ValueError("The model does not support feature importances.")

# Main function to generate 6 plots
def plot_feature_impact_with_top_10(model, X, y, top_n=5):
    """
    Generate and save:
    - Box-and-whisker plots for categorical features
    - Partial dependence plots for continuous features
    - A bar plot for the top 10 most important features

    Parameters:
    -----------
    model : Trained model
        The fitted model with feature importance or partial dependence capability.
    X : pd.DataFrame
        Feature dataset (Test or Train set).
    y : pd.Series or np.array
        Target variable.
    top_n : int
        Number of top features to analyze individually.
    """
    # Ensure /results directory exists
    os.makedirs(results_dir, exist_ok=True)

    if hasattr(model, 'feature_importances_'):
        # Identify top features by importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        top_features = feature_importances.head(top_n)['Feature'].values
    else:
        raise ValueError("The model does not support feature importances.")

    # Generate individual plots for top_n features
    for feature in top_features:
        feature_data = X[feature]
        is_categorical = is_categorical_feature(feature_data)

        if is_categorical:
            # Plot and save box-and-whisker plot for categorical feature
            plot_box_whisker_for_categorical(X, y, feature, results_dir)
        else:
            # Plot partial dependence for continuous feature
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                features=[feature],
                grid_resolution=50,
                ax=ax
            )
            display.axes_[0, 0].set_ylabel("Price (Reverse Log Scale)")
            yticks = display.axes_[0, 0].get_yticks()
            display.axes_[0, 0].set_yticklabels([round(reverse_log_transform(y), 2) for y in yticks])
            
            # Save plot
            plot_path = os.path.join(results_dir, f"{feature}_continuous_partial_dependence.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Partial dependence plot for {feature} saved to {plot_path}")

# Generate the top 10 feature importance plot
plot_top_10_features(best_model, X, results_dir)

# Call the function with updated top 10 feature importance logic
plot_feature_impact_with_top_10(best_model, X_test, y_test, top_n=5)
