import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'flaredataset.csv'
df = pd.read_csv(file_path)

# Handle missing values: choose one method
# Option 1: Drop rows with missing values
# df = df.dropna()

# Option 2: Fill missing values with a constant
#df = df.fillna(0)  # You can also use 'unknown' for categorical columns if necessary

# Option 3: Fill missing numeric values with the column mean, and categorical with the mode
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

# Convert boolean values to integers (0 and 1)
df_encoded = df_encoded.astype(int)

# Save the processed dataset
df_encoded.to_csv('conv.csv', index=False)

print(f"Processed dataset saved to: {'conv.csv'}")
