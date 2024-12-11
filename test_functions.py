import pandas as pd
import os

def test_data_file_exists():
    """Test if the raw data file exists in the expected location."""
    file_path = "data/raw/AB_NYC_2019.csv"
    assert os.path.exists(file_path), f"Data file not found at {file_path}!"

def test_data_loading():
    """Test if the data loads correctly and contains required columns."""
    data = pd.read_csv("data/raw/AB_NYC_2019.csv")
    required_columns = {'id', 'name', 'host_id', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price'}
    assert set(required_columns).issubset(data.columns), f"Missing required columns: {required_columns - set(data.columns)}"

def test_price_range():
    """Test if rental prices are within a reasonable range."""
    data = pd.read_csv("data/raw/AB_NYC_2019.csv")
    assert (data['price'] >= 0).all(), "Prices contain negative values!"
    assert (data['price'] <= 10000).all(), "Prices contain unreasonably high values!"

def test_missing_values():
    """Test if critical columns have missing values."""
    data = pd.read_csv("data/raw/AB_NYC_2019.csv")
    critical_columns = ['price', 'neighbourhood_group', 'room_type']
    for column in critical_columns:
        assert data[column].isnull().sum() == 0, f"Missing values found in column {column}!"

def test_unique_ids():
    """Test if all listing IDs are unique."""
    data = pd.read_csv("data/raw/AB_NYC_2019.csv")
    assert data['id'].is_unique, "Listing IDs are not unique!"
