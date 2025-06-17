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

