import pandas as pd
from src.utils import encode_categorical, split_data

def test_encode_categorical():
    df = pd.DataFrame({
        'A': ['cat', 'dog', 'cat'],
        'B': ['red', 'blue', 'red']
    })
    categorical_cols = ['A', 'B']
    df_encoded, encoders = encode_categorical(df, categorical_cols)
    # Check that columns are encoded as integers
    assert df_encoded['A'].dtype == 'int32' or df_encoded['A'].dtype == 'int64'
    assert df_encoded['B'].dtype == 'int32' or df_encoded['B'].dtype == 'int64'
    # Check that encoders are returned
    assert set(encoders.keys()) == set(categorical_cols)

def test_split_data():
    df = pd.DataFrame({'x': range(10)})
    X = df
    y = pd.Series([0,1]*5)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42)
    # Check sizes
    assert len(X_train) == 6
    assert len(X_val) == 2
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_val) == 2
    assert len(y_test) == 2
