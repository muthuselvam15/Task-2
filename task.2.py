import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
gender_df = pd.read_csv("gender_submission.csv")
print("\n===== TRAIN DATASET (FIRST 5 ROWS) =====")
print(train_df.head())
print("\n===== TEST DATASET (FIRST 5 ROWS) =====")
print(test_df.head())
print("\n===== GENDER SUBMISSION DATASET =====")
print(gender_df.head())
print("\n===== TRAIN DATASET INFO =====")
print(train_df.info())
print("\n===== MISSING VALUES (TRAIN DATASET) =====")
print(train_df.isnull().sum())
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)
print("\n===== MISSING VALUES AFTER CLEANING (TRAIN) =====")
print(train_df.isnull().sum())
print("\n===== DESCRIPTIVE STATISTICS =====")
print(train_df.describe())
sns.set(style="whitegrid")
plt.figure(figsize=(5,4))
sns.countplot(x='Survived', data=train_df)
plt.title("Survival Count")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title("Survival by Gender")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title("Survival by Passenger Class")
plt.show()

# ---- Age Distribution ----
plt.figure(figsize=(6,4))
sns.histplot(train_df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# ---- Fare Distribution ----
plt.figure(figsize=(6,4))
sns.histplot(train_df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# ---- Fare vs Survival ----
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Fare', data=train_df)
plt.title("Fare vs Survival")
plt.show()

# ---- Embarked vs Survival ----
plt.figure(figsize=(6,4))
sns.countplot(x='Embarked', hue='Survived', data=train_df)
plt.title("Embarked vs Survival")
plt.show()

# ---------------------------------------------------------
# 7. Correlation Analysis
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
corr = train_df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------
# 8. Key Insights
# ---------------------------------------------------------
print("\n===== KEY INSIGHTS =====")
print("1. Female passengers had a significantly higher survival rate.")
print("2. First class passengers survived more than second and third class.")
print("3. Passengers who paid higher fares had better survival chances.")
print("4. Age had moderate influence, while gender and class were dominant factors.")
print("5. Passengers embarking from Cherbourg (C) had higher survival.")

print("\n===== TASK-02 COMPLETED SUCCESSFULLY =====")
