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



   
