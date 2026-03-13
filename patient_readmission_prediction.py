import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("readmission_predictions_final.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

LabelEncoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = LabelEncoder.fit_transform(df[column])


plt.hist(df['age'], bins=20)    
plt.title("Age distribution of patients")
plt.xlabel("Age")
plt.ylabel("Number of patients")
plt.show()

X = df.drop("actual_readmission", axis=1)
y = df["actual_readmission"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)




    



