# Titanic Dataset - Data Cleaning & Preprocessing
This project is part of the AI & ML Internship Task 1. The goal is to clean and preprocess the Titanic dataset for machine learning purposes.

## Task Objective
- Handle missing values
- Convert categorical data into numerical data
- Normalize/standardize numerical columns
- Detect and remove outliers

## Tools Used
- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

## Files in This Repo
- `task1_titanic_cleaning.py` – Python script for preprocessing the Titanic dataset
- `titanic.csv` – Dataset file
- `README.md` – Description and summary of the task

## 🔍 Steps Performed
1. Loaded the dataset using pandas
2. Explored data types and missing values
3. Filled in missing values in `Age` and `Embarked`
4. Dropped the `Cabin` column (too many nulls)
5. Encoded categorical variables:
   - `Sex`: Label Encoding
   - `Embarked`: One-Hot Encoding
6. Scaled numerical columns `Age` and `Fare` using StandardScaler
7. Visualized outliers using boxplots and removed them with IQR method
8. Exported the clean dataset (ready for ML)

## Author
Ajeesha A  

## Note
This is a beginner-level task to understand basic data preprocessing before training ML models.

