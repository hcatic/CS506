import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    """Drop unnecessary columns."""
    df_cleaned = df.drop(columns=[
        'id', 'host id', 'NAME', 'host name', 'license', 'country', 'country code', 'house_rules'
    ])
    return df_cleaned

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df['reviews per month'] = df['reviews per month'].fillna(0)
    df['last review'] = pd.to_datetime(df['last review'], errors='coerce')
    earliest_date = df['last review'].min()
    df['last review'] = df['last review'].fillna(earliest_date)

    # Fill numeric and categorical columns
    num_columns = df.select_dtypes(include=['float64']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    int_columns = ['minimum nights', 'number of reviews', 'review rate number', 'calculated host listings count']
    for col in int_columns:
        df[col] = df[col].astype(int)

    # Clean price and service fee
    df['price'] = df['price'].astype(str).replace(r'[\$,]', '', regex=True).astype(float)
    df['service fee'] = df['service fee'].astype(str).replace(r'[\$,]', '', regex=True).astype(float)

    return df

def encode_data(df):
    """Encode categorical variables."""
    small_categorical_cols = ['host_identity_verified', 'room type', 'cancellation_policy', 'instant_bookable', 'neighbourhood group']
    df = pd.get_dummies(df, columns=small_categorical_cols, drop_first=True)

    le = LabelEncoder()
    df['neighbourhood_encoded'] = le.fit_transform(df['neighbourhood'])
    df.drop('neighbourhood', axis=1, inplace=True)
    return df

def transform_data(df):
    """Apply log transformations and standardization to reduce skew and scale data."""
    df['minimum nights'] = np.log1p(df['minimum nights'])
    df['availability 365'] = np.log1p(df['availability 365'])
    df['number of reviews'] = np.log1p(df['number of reviews'])

    for col in ['minimum nights', 'availability 365', 'number of reviews']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())

    numerical_cols = ['service fee', 'minimum nights', 'availability 365', 'number of reviews', 'neighbourhood_encoded']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df