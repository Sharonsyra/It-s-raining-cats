import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import encode_categorical, split_data
from data_processing import load_data, clean_data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, 'data', 'cat_breeds_clean.csv')
model_path = os.path.join(BASE_DIR, 'models', 'cat_breed_model.pkl')
encoders_path = os.path.join(BASE_DIR, 'models', 'cat_breed_encoders.pkl')

df = load_data(file_path)
df = clean_data(df)

cat_cols = ['Gender', 'Neutered_or_spayed', 'Fur_colour_dominant', 'Fur_pattern', 'Eye_colour', 'Allowed_outdoor', 'Preferred_food', 'Country']
df_encoded, encoders = encode_categorical(df, cat_cols)

X = df_encoded.drop('Breed', axis=1)
y = df_encoded['Breed']

# Split data: 60% train, 20% validation, 20% test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_size=0.6, val_size=0.2, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

val_score = model.score(X_val, y_val)
print(f'Validation Accuracy: {val_score:.4f}')

joblib.dump(model, model_path)
joblib.dump(encoders, encoders_path)
