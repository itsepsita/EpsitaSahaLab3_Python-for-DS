import pandas as pd
from src.data_preprocessing import load_data, handle_missing_values, encode_columns
from src.feature_engineering import split_data
from src.model_training import evaluate_all_models
from src.utils import plot_confusion_matrix

# Load and preprocess the data
data = load_data('data/heart_disease_data.csv')
data = handle_missing_values(data)
data = encode_columns(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(data)

# Evaluate models
results = evaluate_all_models(X_train, X_test, y_train, y_test)

# Print results for all models
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['Accuracy']}")
    print(f"Confusion Matrix:\n{result['Confusion Matrix']}")
    print(f"Classification Report:\n{result['Classification Report']}")

    # Plot confusion matrix
    plot_confusion_matrix(result['Confusion Matrix'], labels=['No', 'Yes'])
