
"""
Earthquake Dataset – Data Analysis Script
-----------------------------------------

This script performs data analysis for the following tasks:

- Feature analysis
- Correlation matrix + heatmap
- Independence / redundancy analysis
- Data statistics
- Distribution plots (histograms, boxplots)
- Feature importance using Mutual Information (classification & regression)
- Scatter plots for spatial and relational patterns

NOTE:
Place the dataset file path below (CSV file downloaded from Kaggle).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------

DATA_PATH = "earthquake_1995-2023.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------------
# 2. FEATURE ANALYSIS
# ---------------------------------------------------------------------

print("\n=== Feature Types ===")
print(df.dtypes)

print("\n=== Missing Values per Column ===")
print(df.isna().sum())

# ---------------------------------------------------------------------
# 3. DATA STATISTICS
# ---------------------------------------------------------------------

print("\n=== Numerical Feature Statistics ===")
print(df.describe())

# ---------------------------------------------------------------------
# 4. CORRELATION MATRIX (Numerical Features Only)
# ---------------------------------------------------------------------

num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap="viridis")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(num_df.columns)), num_df.columns, rotation=90)
plt.yticks(range(len(num_df.columns)), num_df.columns)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 5. HISTOGRAMS OF NUMERICAL FEATURES
# ---------------------------------------------------------------------

for column in num_df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df[column].dropna(), bins=40)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 6. BOXPLOTS FOR OUTLIER ANALYSIS
# ---------------------------------------------------------------------

for column in num_df.columns:
    plt.figure(figsize=(5, 4))
    plt.boxplot(df[column].dropna())
    plt.title(f"Boxplot of {column}")
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Tsunami Classification)
# ---------------------------------------------------------------------

if 'tsunami' in df.columns:
    print("\n=== Feature Importance for Tsunami Prediction (Mutual Information) ===")

    y_class = df['tsunami']
    X_class = df.drop(columns=['tsunami'])

    # Encode categorical features
    for col in X_class.select_dtypes(include=['object']).columns:
        X_class[col] = LabelEncoder().fit_transform(X_class[col].astype(str))

    mi_scores = mutual_info_classif(X_class.fillna(0), y_class)

    for feat, score in zip(X_class.columns, mi_scores):
        print(f"{feat}: {score}")

# ---------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Significance Regression)
# ---------------------------------------------------------------------

if 'sig' in df.columns:
    print("\n=== Feature Importance for Significance Regression (Mutual Information) ===")

    y_reg = df['sig']
    X_reg = df.drop(columns=['sig'])

    # Encode categorical features
    for col in X_reg.select_dtypes(include=['object']).columns:
        X_reg[col] = LabelEncoder().fit_transform(X_reg[col].astype(str))

    mi_reg = mutual_info_regression(X_reg.fillna(0), y_reg)

    for feat, score in zip(X_reg.columns, mi_reg):
        print(f"{feat}: {score}")

# ---------------------------------------------------------------------
# 9. SCATTER PLOTS (GEOGRAPHICAL)
# ---------------------------------------------------------------------

if 'latitude' in df.columns and 'longitude' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['longitude'], df['latitude'], s=2)
    plt.title("Earthquake Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 10. DEPTH VS MAGNITUDE (Colored by Tsunami)
# ---------------------------------------------------------------------

if 'depth' in df.columns and 'magnitude' in df.columns and 'tsunami' in df.columns:
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(df['depth'], df['magnitude'], c=df['tsunami'], s=4)
    plt.colorbar(scatter, label="Tsunami (0/1)")
    plt.title("Depth vs Magnitude (Colored by Tsunami)")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()

print("\nAnalysis complete.")
