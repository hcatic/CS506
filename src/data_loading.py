import pandas as pd

def load_data(filepath):
    """Load the Airbnb dataset from the specified file."""
    df = pd.read_csv(filepath)
    return df

def show_basic_info(df):
    """Display head and info about the DataFrame."""
    print(df.head())
    print(df.info())

def summary_statistics(df):
    """Display summary statistics of the DataFrame."""
    return df.describe()