import pandas as pd

def add_property_age(df):
    """Calculate and add property_age based on Construction year."""
    current_year = pd.Timestamp.now().year
    df['property_age'] = current_year - df['Construction year']
    df['property_age'] = df['property_age'].apply(lambda x: x if x > 0 else 0)
    df.drop('Construction year', axis=1, inplace=True)
    return df

def bin_property_age(df):
    """Bin the property_age into categories and one-hot encode."""
    max_age = df['property_age'].max()
    bins = [0, 5, 10, 15, max_age + 1]
    labels = ['0-5 years', '6-10 years', '11-15 years', '16+ years']
    df['age_category'] = pd.cut(df['property_age'], bins=bins, labels=labels, include_lowest=True)
    df = pd.get_dummies(df, columns=['age_category'], drop_first=True)
    df.drop('property_age', axis=1, inplace=True)
    return df

def days_since_last_review(df):
    """Calculate days since last review."""
    df['days_since_last_review'] = (pd.Timestamp('today') - df['last review']).dt.days
    max_days = df['days_since_last_review'].max()
    df['days_since_last_review'] = df['days_since_last_review'].fillna(max_days)
    df.drop('last review', axis=1, inplace=True)
    return df

def create_interaction_terms(df):
    """Create interaction terms between neighbourhood groups and private rooms."""
    neighbourhood_group_cols = [col for col in df.columns if col.startswith('neighbourhood group_')]
    if 'room type_Private room' in df.columns:
        for col in neighbourhood_group_cols:
            interaction_term = f"{col}_PrivateRoom"
            df[interaction_term] = df[col] * df['room type_Private room']
    return df
