import pandas as pd

def load_data(path, sep=';'):
    return pd.read_csv(path, sep=sep)

def clean_data(df):
    # Example cleaning steps
    df = df.copy()
    # Fix breed names
    df['Breed'] = df['Breed'].replace({'Ankora': 'Angora', 'angora': 'Angora'})
    # Standardize country names
    df['Country'] = df['Country'].str.strip().str.capitalize()
    # Remove negative ages
    df = df[df['Age_in_years'] >= 0]
    # Fill missing values
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Drop rows with too many missing values (if needed)
    df = df.dropna(thresh=int(0.8 * len(df.columns)))
    return df
