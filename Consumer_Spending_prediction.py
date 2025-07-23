
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("CC GENERAL.csv")
df.head()

print("Missing values:\n", df.isnull().sum())
df = df.dropna()

df.rename(columns={"TENURE": "Tenure", "CREDIT_LIMIT": "Credit_Limit"}, inplace=True)

df['High_Spender'] = df['PURCHASES'].apply(lambda x: 1 if x > df['PURCHASES'].median() else 0)

df.drop(['CUST_ID'], axis=1, inplace=True)

X = df.drop(['High_Spender'], axis=1)
y = df['High_Spender']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
plt.title("Feature Importance for High Spender Classification")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
