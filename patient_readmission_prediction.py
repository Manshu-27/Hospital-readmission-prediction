import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Hospital_readmission.csv")

print("First 5 Rows:")
print(df.head())

Print("\nDataset Info:")
print(df.Info())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

label_Encoder = labelEncoder()

For column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

target_column = "readmitted"

X = df.drop(target_column, axis=1)
y = df[target_column]

X_tarin, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuarcy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(calssificaton_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("confusion Matrix")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

print("\nConfusion Matrix:")
print(cm)



    


