import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    return df

def convert_data_types(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y-%m-%d %H:%M:%S')
    df['is_fraud'] = df['is_fraud'].astype(int)
    return df

def feature_engineering(df):
    df['transaction_hour'] = df['transaction_date'].dt.hour
    df['transaction_day'] = df['transaction_date'].dt.day
    df['transaction_month'] = df['transaction_date'].dt.month
    df['transaction_dayofweek'] = df['transaction_date'].dt.dayofweek
    df['transaction_weekend'] = df['transaction_dayofweek'].isin([5,6]).astype(int)
    df['transaction_amount_log'] = np.log1p(df['transaction_amount'])
    return df

def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = onehot.fit_transform(df[categorical_columns])
    encoded_feature_names = onehot.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    return df, onehot

def scale_numerical_features(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns = numerical_columns.drop('is_fraud') if 'is_fraud' in numerical_columns else numerical_columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df

def split_data(df):
    X = df.drop(['is_fraud', 'transaction_date'], axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_data(file_path):
    df = load_and_inspect_data(file_path)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = feature_engineering(df)
    df, onehot_encoder = encode_categorical_variables(df)
    df = remove_correlated_features(df)
    df, scaler = scale_numerical_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    return df, X_train, X_test, y_train, y_test, onehot_encoder, scaler

if __name__ == "__main__":
    file_path = "filtered_transactions.csv"
    df, X_train, X_test, y_train, y_test, onehot_encoder, scaler = preprocess_data(file_path)
    
    print("\nPreprocessing completed.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    df.to_csv("preprocessed_data.csv", index=False)
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    
    print("\nPreprocessed data and objects saved.")
