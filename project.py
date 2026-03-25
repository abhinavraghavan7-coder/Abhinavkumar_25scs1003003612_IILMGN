import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("dataset.csv")

print("📊 Dataset Preview:\n")
print(data.head())

# =========================
# FEATURES & LABEL
# =========================
X = data[['Hours', 'Sleep', 'Previous']]
y = data['Marks']

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================
# MODEL TRAINING
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

# =========================
# MODEL ACCURACY
# =========================
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print(f"\n📈 Model Accuracy (R2 Score): {accuracy:.2f}")

# =========================
# USER INPUT
# =========================
print("\n🔹 Enter Student Details 🔹")

hours = float(input("Study Hours: "))
sleep = float(input("Sleep Hours: "))
previous = float(input("Previous Marks: "))

# =========================
# PREDICTION
# =========================
prediction = model.predict([[hours, sleep, previous]])

print(f"\n🎯 Predicted Marks: {prediction[0]:.2f}")

# =========================
# GRAPH VISUALIZATION
# =========================
plt.scatter(data['Hours'], data['Marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

# =========================
# EXTRA INSIGHTS
# =========================
print("\n📌 Model Coefficients:")
print(f"Hours Impact: {model.coef_[0]:.2f}")
print(f"Sleep Impact: {model.coef_[1]:.2f}")
print(f"Previous Marks Impact: {model.coef_[2]:.2f}")

print(f"\n📊 Intercept: {model.intercept_:.2f}")

print("\n🔥 Project Completed Successfully!")