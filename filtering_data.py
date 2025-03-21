import pandas as pd

# Load the transaction data
df = pd.read_csv('transactions_train.csv')

# Filter fraudulent transactions
df_fraud = df[df['is_fraud'] == 1]

# Filter 10 or 11 non-fraudulent transactions
df_non_fraud = df[df['is_fraud'] == 0].sample(n=20, random_state=42)

# Combine the two dataframes
df_filtered = pd.concat([df_fraud, df_non_fraud], ignore_index=True)

# Save the filtered data to a new CSV file
df_filtered.to_csv('filtered_transactions.csv', index=False)

print('Filtered data saved to filtered_transactions.csv')
