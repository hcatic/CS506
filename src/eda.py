import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

def show_data_overview(df):
    """
    Print basic information about the DataFrame:
    - DataFrame info (structure, data types)
    - Head of the DataFrame
    - Summary statistics for numerical columns
    - Missing values count
    """
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 Rows:")
    display(df.head())
    
    print("\nSummary Statistics:")
    display(df.describe())
    
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values)

def ensure_price_numeric(df):
    """
    Ensure 'price' is numeric. If non-numeric values exist, convert them to NaN
    and drop those rows.
    """
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    return df

def plot_rental_price_distribution(df, max_price=None):
    """
    Plot the distribution of rental prices.
    If max_price is provided, cap the plotted data to that value for better visualization.
    """
    data = df['price'] if max_price is None else df[df['price'] <= max_price]['price']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=50, kde=True)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rental Prices')
    plt.show()

def plot_room_type_distribution(df):
    """
    Plot the distribution of room types.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x='room type', data=df)
    plt.xlabel('Room Type')
    plt.ylabel('Count')
    plt.title('Distribution of Room Types')
    plt.show()

def plot_neighbourhood_group_distribution(df):
    """
    Plot the distribution of listings per neighbourhood group.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(y='neighbourhood group', data=df, order=df['neighbourhood group'].value_counts().index)
    plt.xlabel('Count')
    plt.ylabel('Neighbourhood Group')
    plt.title('Listings per Neighbourhood Group')
    plt.show()

def plot_price_by_room_type(df, max_price=None):
    """
    Plot a boxplot of price distribution by room type.
    If max_price is provided, set the y-limit for better visualization.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='room type', y='price', data=df)
    plt.xlabel('Room Type')
    plt.ylabel('Price')
    plt.title('Price Distribution by Room Type')
    if max_price is not None:
        plt.ylim(0, max_price)
    plt.show()

def plot_price_by_neighbourhood_group(df, max_price=None):
    """
    Plot a boxplot of price distribution by neighbourhood group.
    If max_price is provided, set the y-limit for better visualization.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='neighbourhood group', y='price', data=df)
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Price')
    plt.title('Price Distribution by Neighbourhood Group')
    if max_price is not None:
        plt.ylim(0, max_price)
    plt.show()

def plot_availability_by_room_type(df):
    """
    Plot availability distribution (365 days) by room type.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='room type', y='availability 365', data=df)
    plt.xlabel('Room Type')
    plt.ylabel('Availability (days per year)')
    plt.title('Availability Distribution by Room Type')
    plt.show()

def plot_price_vs_number_of_reviews(df, max_price=None):
    """
    Plot a scatterplot of price vs. number of reviews.
    If max_price is provided, filter the data.
    """
    data = df if max_price is None else df[df['price'] <= max_price]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='number of reviews', y='price', data=data, alpha=0.5)
    plt.xlabel('Number of Reviews')
    plt.ylabel('Price')
    plt.title('Price vs. Number of Reviews')
    plt.show()

def create_price_map(df, output_path='nyc_airbnb_price_map.html'):
    """
    Create a Folium map of listings, colored by price category:
    - Blue if price < 100
    - Orange if 100 <= price < 300
    - Red if price >= 300

    Saves the map as an HTML file.
    """
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    for _, row in df.iterrows():
        if row['price'] < 100:
            color = 'blue'
        elif row['price'] < 300:
            color = 'orange'
        else:
            color = 'red'

        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Price: ${row['price']}, Room Type: {row['room type']}"
        ).add_to(nyc_map)

    nyc_map.save(output_path)
    print(f"Map saved as {output_path}")
