import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath, target_column='Churn', test_size=0.2, random_state=42):
    """
    Loads the dataset, performs preprocessing, and splits into train/test sets.
    
    Args:
        filepath (str): Path to the CSV file.
        target_column (str): Name of the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test: Split and preprocessed data.
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Handle missing values in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df[target_column] = le_target.fit_transform(df[target_column])
    
    # Split into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve feature names
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    return X_train, X_test, y_train, y_test
