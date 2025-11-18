import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, categorical_cols):
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

def split_data(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_size), random_state=random_state
    )
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
