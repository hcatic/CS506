from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_and_evaluate_models(df, target='price'):
    """Train and evaluate multiple regression models."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model: {name}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R^2 Score: {r2:.2f}\n")

def cross_validate_models(df, target='price', cv_folds=5):
    """Perform cross-validation on multiple models."""
    X = df.drop(columns=[target])
    y = df[target]

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        cv_mse = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_mae = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
        cv_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

        mean_mse = -np.mean(cv_mse)
        mean_mae = -np.mean(cv_mae)
        mean_r2 = np.mean(cv_r2)

        print(f"Model: {name}")
        print(f"Cross-validated Mean Squared Error: {mean_mse:.2f}")
        print(f"Cross-validated Mean Absolute Error: {mean_mae:.2f}")
        print(f"Cross-validated R^2 Score: {mean_r2:.2f}\n")
