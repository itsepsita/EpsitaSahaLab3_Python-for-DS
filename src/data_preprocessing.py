

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Loads dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def handle_missing_values(data):
    """
    Handle missing values by dropping rows with missing values.
    """
    return data.dropna()

def encode_columns(data):
    """
    Label encodes categorical columns and one-hot encodes others.
    """
    label_encoder = LabelEncoder()
    
    # Label encode categorical columns
    data['AgeCategory'] = label_encoder.fit_transform(data['AgeCategory'])
    data['Race'] = label_encoder.fit_transform(data['Race'])
    data['GenHealth'] = label_encoder.fit_transform(data['GenHealth'])
    
    # One-hot encode other categorical columns
    data = pd.get_dummies(data, drop_first=True)
    
    return data
