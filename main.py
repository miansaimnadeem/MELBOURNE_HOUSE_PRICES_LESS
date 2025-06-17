import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Current working directory:", os.getcwd())
try:
	df = pd.read_csv(r'F:\Universty\4th Semester\Intro to Ds\MELBOURNE_HOUSE_PRICES_LESS.csv')
except FileNotFoundError:
	print("Error: The file 'Electric_Vehicle_Population_Data.csv' was not found. Please check the file path and ensure the file exists in the directory above.")
	exit()
 
print("Initial Data Info:")
print(df.info())

print("\nFirst 5 rows")
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values before Feature Engineering:")
print(df.isnull().sum())

# missing values

df.dropna(subset=['Price'], inplace=True)

df['Date']=pd.to_datetime(df['Date'], format='%d/%m/%Y') 
df['Year'] = df['Data'].dt.year
df['Month'] = df['Data'].dt.month
df['Day'] = df['Data'].dt.day
df['DayOfWeek']=df['Data'].dt.dayofweek

df.drop(['Date','Address'], axis=1, inplace=True)

print("\nMissing Values after handling 'Price' and Feature Engineering 'Data':")
print(df.isnull().sum())


categorical_cols = df.select_dtypes(include='object').columns
print(f"\nCategorical columns before encoding: {categorical_cols.tolist()}")


print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")


df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nFirst 5 rows after feature engineering and encoding:")
print(df_encoded.head())

print("\nInfo after feature engineering and encoding:")
print(df_encoded.info())

df_encoded.to_csv('melbourne_housing_processed.csv', index=False)



# Set style for plots
sns.set_style("whitegrid")

# 1. Distribution of the target variable (Price)
plt.figure(figsize=(10, 6))
sns.histplot(df_encoded['Price'], kde=True, bins=50)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('price_distribution.png')

# 2. Distributions of key numerical features
numerical_features = ['Rooms', 'Postcode', 'Propertycount', 'Distance', 'Year', 'Month', 'Day', 'DayOfWeek']

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df_encoded[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('numerical_features_distributions.png')

# 3. Correlation Matrix (focus on numerical and new date features)
# Select only numerical columns for correlation matrix
numerical_df = df_encoded[numerical_features + ['Price']]
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features and Price')
plt.savefig('correlation_matrix.png')

# 4. Relationships between key categorical features and Price (using original df before one-hot encoding for easier plotting)
# Reload original df to easily access original categorical columns for plotting
df_original = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
df_original.dropna(subset=['Price'], inplace=True) # Ensure 'Price' is not missing

categorical_for_plotting = ['Type', 'Method', 'Regionname']

plt.figure(figsize=(18, 6))
for i, col in enumerate(categorical_for_plotting):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=col, y='Price', data=df_original, palette='viridis')
    plt.title(f'Price by {col}')
    plt.xlabel(col)
    plt.ylabel('Price')
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout()
plt.savefig('categorical_features_price_boxplot.png')

print("EDA and Feature Engineering complete. Plots saved as .png files and processed data saved as melbourne_housing_processed.csv.")

   
