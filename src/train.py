import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import encode_categorical, split_data
from data_processing import load_data, clean_data

df = load_data('../data/raw/cat_breeds_clean.csv')
df = clean_data(df)

cat_cols = ['Gender', 'Neutered_or_spayed', 'Fur_colour_dominant', 'Fur_pattern', 'Eye_colour', 'Allowed_outdoor', 'Preferred_food', 'Country']
df_encoded, encoders = encode_categorical(df, cat_cols)

X = df_encoded.drop('Breed', axis=1)
y = df_encoded['Breed']

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

val_score = model.score(X_val, y_val)
print(f'Validation Accuracy: {val_score:.4f}')

joblib.dump(model, '../models/cat_breed_model.pkl')
