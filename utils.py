import pandas as pd
import numpy as np

def load_data():
    """
    Loads the Boston Housing dataset from a public URL and returns it as a pandas DataFrame.
    The dataset is created manually as it's deprecated from scikit-learn.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # now we split this into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # These are the Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # here MEDV is our target variable
    return df

if __name__ == "__main__":
    # This block is for testing the load_data function independently
    housing_df = load_data()
    print("Dataset loaded successfully!")
    print("Shape of the dataset:", housing_df.shape)
    print("\nFirst 5 rows:")
    print(housing_df.head())
    print("\nInfo:")
    housing_df.info()