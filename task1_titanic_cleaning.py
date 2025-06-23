
# Titanic Data Cleaning & Preprocessing - Task 1

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
df = pd.read_csv("titanic.csv")  # Make sure this file is in the same directory
print("First 5 rows:")
print(df.head())

# 3. Explore Dataset
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())

# 4. Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# 5. Encode Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 6. Scale Numerical Features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 7. Visualize and Remove Outliers
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# 8. Final Dataset Check
print("\nCleaned Dataset Info:")
print(df.info())
print("\nFirst 5 Rows of Cleaned Data:")
print(df.head())
