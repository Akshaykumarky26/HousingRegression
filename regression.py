import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Import the load_data function from utils.py
from utils import load_data

def train_evaluate_regression_models():
    """
    Loads the Boston Housing dataset, splits it, trains multiple regression models,
    and evaluates their performance using MSE and R2.
    This function is for initial model evaluation without tuning.
    """
    print("Starting initial regression model training and evaluation (without tuning)...")

    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MSE': mse, 'R2': r2}

        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

    print("\n--- Performance Comparison Report (Initial Models) ---")
    print(f"{'Model':<25} {'MSE':<10} {'R2':<10}")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['MSE']:<10.4f} {metrics['R2']:<10.4f}")
    print("-" * 45)
    return results # Return results for potential comparison later


def tune_and_evaluate_models():
    """
    Loads the Boston Housing dataset, splits it, performs hyperparameter tuning
    for multiple regression models using GridSearchCV, and evaluates their performance.
    """
    print("\nStarting hyperparameter tuning and evaluation...")

    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their parameter grids for GridSearchCV
    # Ensure at least 3 hyperparameters where applicable
    tuned_models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False], # Hyperparameter 1
                'copy_X': [True, False],        # Hyperparameter 2
                'n_jobs': [None, -1]            # Hyperparameter 3 (can be None or -1 for parallel processing)
            }
        },
        'Decision Tree Regressor': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [None, 5, 10, 15], # Hyperparameter 1
                'min_samples_split': [2, 5, 10], # Hyperparameter 2
                'min_samples_leaf': [1, 2, 4],   # Hyperparameter 3
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error'] # Hyperparameter 4
            }
        },
        'Random Forest Regressor': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150], # Hyperparameter 1
                'max_depth': [None, 10, 20],   # Hyperparameter 2
                'min_samples_split': [2, 5, 10], # Hyperparameter 3
                'min_samples_leaf': [1, 2, 4]    # Hyperparameter 4
            }
        }
    }

    tuned_results = {}

    for name, config in tuned_models.items():
        print(f"\n--- Tuning {name} ---")
        # GridSearchCV performs cross-validation on the training set
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=5, # 5-fold cross-validation
            scoring='neg_mean_squared_error', # Optimize for lower MSE
            n_jobs=-1, # Use all available CPU cores
            verbose=1 # Print progress
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred_tuned = best_model.predict(X_test)

        mse_tuned = mean_squared_error(y_test, y_pred_tuned)
        r2_tuned = r2_score(y_test, y_pred_tuned)

        tuned_results[name] = {
            'Best Params': best_params,
            'MSE': mse_tuned,
            'R2': r2_tuned
        }

        print(f"Best parameters for {name}: {best_params}")
        print(f"{name} (Tuned) - MSE: {mse_tuned:.4f}, R2: {r2_tuned:.4f}")

    print("\n--- Performance Comparison Report (Tuned Models) ---")
    print(f"{'Model':<25} {'MSE':<10} {'R2':<10} {'Best Parameters':<50}")
    print("-" * 95)
    for name, metrics in tuned_results.items():
        print(f"{name:<25} {metrics['MSE']:<10.4f} {metrics['R2']:<10.4f} {str(metrics['Best Params']):<50}")
    print("-" * 95)
    return tuned_results


if __name__ == "__main__":
    # Run initial evaluation
    initial_results = train_evaluate_regression_models()

    # Run hyperparameter tuning and evaluation
    tuned_results = tune_and_evaluate_models()

    # Optional: Compare initial vs. tuned results directly
    print("\n--- Overall Performance Comparison (Initial vs. Tuned) ---")
    print(f"{'Model':<25} {'Initial MSE':<15} {'Tuned MSE':<15} {'Initial R2':<15} {'Tuned R2':<15}")
    print("-" * 90)
    for name in initial_results.keys():
        initial_mse = initial_results[name]['MSE']
        tuned_mse = tuned_results[name]['MSE']
        initial_r2 = initial_results[name]['R2']
        tuned_r2 = tuned_results[name]['R2']
        print(f"{name:<25} {initial_mse:<15.4f} {tuned_mse:<15.4f} {initial_r2:<15.4f} {tuned_r2:<15.4f}")
    print("-" * 90)