import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Create screenshots folder
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Load Dataset
df = pd.read_csv("car_data.csv")

print("First 5 Records:")
print(df.head())

# Dataset Preview Image
plt.figure(figsize=(8,2))
plt.axis('off')

table = plt.table(
    cellText=df.head().values,
    colLabels=df.columns,
    loc='center'
)

plt.savefig(
    "screenshots/dataset_preview.png",
    bbox_inches='tight'
)
plt.close()

# Convert Categorical Data
df = pd.get_dummies(
    df,
    columns=['Fuel_Type'],
    drop_first=True
)

# Features and Target
X = df.drop(
    ['Car_Name', 'Selling_Price'],
    axis=1
)

y = df['Selling_Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nR2 Score:", round(r2,2))
print("Mean Absolute Error:", round(mae,2))

# Correlation Heatmap
plt.figure(figsize=(8,5))

sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.savefig(
    "screenshots/correlation_heatmap.png"
)

plt.close()

# Actual vs Predicted
plt.figure(figsize=(6,4))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

plt.savefig(
    "screenshots/actual_vs_predicted.png"
)

plt.close()

# Model Performance
plt.figure(figsize=(5,3))

plt.bar(
    ["R2 Score"],
    [r2]
)

plt.title("Model Performance")

plt.savefig(
    "screenshots/model_performance.png"
)

plt.close()

print("\nProject Completed Successfully!")