import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'flaredataset.csv'
df = pd.read_csv(file_path)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':  # For categorical columns
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # For numeric columns
        df[col].fillna(df[col].mean(), inplace=True)

# Clean unnecessary "Unnamed" columns
df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Identify categorical columns
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

# Separate binary and multi-category columns
binary_columns = [col for col in categorical_columns if df_cleaned[col].nunique() == 2]
multi_category_columns = [col for col in categorical_columns if df_cleaned[col].nunique() > 2]

# Apply Label Encoding to binary columns
label_encoders = {}
for col in binary_columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Apply One-Hot Encoding to multi-category columns
df_encoded = pd.get_dummies(df_cleaned, columns=multi_category_columns)

# Mapping of frequencies to days based on the given flare frequencies
frequency_mapping = {
    'Flare Frequency_1 per month': 30,        # 1 flare per month = 30 days
    'Flare Frequency_1 per year': 365,        # 1 flare per year = 365 days
    'Flare Frequency_2 per month': 15,        # 2 flares per month = 15 days
    'Flare Frequency_2 per year': 182.5,      # 2 flares per year = 182.5 days
    'Flare Frequency_3 per month': 10,        # 3 flares per month = 10 days
    'Flare Frequency_4 per year': 91.25,      # 4 flares per year = 91.25 days
    'Flare Frequency_5 per month': 6,         # 5 flares per month = 6 days
}

# Check if relevant Flare Frequency columns are present
days_until_next_flare = 0
for col, days in frequency_mapping.items():
    if col in df_encoded.columns:
        days_until_next_flare += df_encoded[col] * days

df_encoded['Days_Until_Next_Flare'] = days_until_next_flare

# Convert boolean values to integers (0 and 1)
df_encoded = df_encoded.astype(int)

# Save the processed dataset
df_encoded.to_csv('conv3.csv', index=False)

print(f"Processed dataset with 'Days Until Next Flare' saved to: {'conv3.csv'}")
