import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)
    print("Data Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    return df

def handle_missing_values(df):
    # Fill missing values in numerical columns with median
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Fill missing values in categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    # Handle 'payer_mobile_anonymous' separately due to high missing rate
    df['payer_mobile_anonymous'] = df['payer_mobile_anonymous'].fillna('Unknown')
    
    return df

def convert_data_types(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y-%m-%d %H:%M:%S')
    df['is_fraud'] = df['is_fraud'].astype(int)  # Keep as int instead of bool
    return df

def feature_engineering(df):
    df['transaction_hour'] = df['transaction_date'].dt.hour
    df['transaction_day'] = df['transaction_date'].dt.day
    df['transaction_month'] = df['transaction_date'].dt.month
    df['transaction_dayofweek'] = df['transaction_date'].dt.dayofweek
    df['amount_bin'] = pd.cut(df['transaction_amount'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    return df


def encode_categorical_variables(df):
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    return df, le

def scale_numerical_features(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns = numerical_columns.drop('is_fraud') if 'is_fraud' in numerical_columns else numerical_columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

def split_data(df):
    X = df.drop(['is_fraud', 'transaction_date', 'transaction_id_anonymous'], axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def process_anonymized_fields(df):
    anonymized_columns = ['transaction_payment_mode_anonymous', 'payment_gateway_bank_anonymous', 
                          'payer_browser_anonymous', 'payer_email_anonymous', 'payee_ip_anonymous', 
                          'payer_mobile_anonymous', 'transaction_id_anonymous', 'payee_id_anonymous']
    for col in anonymized_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def preprocess_data(file_path):
    # Load and inspect data
    df = load_and_inspect_data(file_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Convert data types
    df = convert_data_types(df)
    
    # Feature engineering
    df = feature_engineering(df)

    # To process_anonymized_fields
    df = process_anonymized_fields(df)

    # Encode categorical variables
    df, label_encoder = encode_categorical_variables(df)
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)
    
    return df, X_train, X_test, y_train, y_test, label_encoder, scaler

if __name__ == "__main__":
    file_path = "transactions_train.csv"
    df, X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(file_path)
    
    print("\nPreprocessing completed.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Save preprocessed data and objects for later use
    df.to_csv("preprocessed_data.csv", index=False)
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    
    print("\nPreprocessed data and objects saved.")
