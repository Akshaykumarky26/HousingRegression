import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np # Import numpy for potential future use or type hinting

# Import the load_data function from utils.py
from utils import load_data

def train_evaluate_regression_models():
    """
    Loads the Boston Housing dataset, splits it, trains multiple regression models,
    and evaluates their performance using MSE and R2.
    """
    print("Starting regression model training and evaluation...")

    # 1. Load the dataset
    df = load_data()
    print("Dataset loaded successfully. Shape:", df.shape)

    # Define features (X) and target (y)
    X = df.drop('MEDV', axis=1) # All columns except 'MEDV' are features
    y = df['MEDV']             # 'MEDV' is the target variable

    # 2. Split the data into training and testing sets
    # We use a fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: Training samples = {len(X_train)}, Testing samples = {len(X_test)}")

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42)
    }

    # Store results
    results = {}

    # 3. Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MSE': mse, 'R2': r2}

        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

    # 4. Print comparison report
    print("\n--- Performance Comparison Report (Initial Models) ---")
    print(f"{'Model':<25} {'MSE':<10} {'R2':<10}")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['MSE']:<10.4f} {metrics['R2']:<10.4f}")
    print("-" * 45)


if __name__ == "__main__":
    train_evaluate_regression_models()