from sklearn.model_selection import train_test_split

def split_data(data, target_column='HeartDisease', test_size=0.3):
    """
    Splits the data into training and testing sets.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=42)
