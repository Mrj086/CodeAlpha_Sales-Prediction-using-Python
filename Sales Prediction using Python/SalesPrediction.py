# ============================================================
# TASK 4: Sales Prediction using Python
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
file_path = (r"C:\Users\USER\Downloads\Advertising.csv")
df = pd.read_csv(file_path)

# ------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Drop unnecessary column if exists
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nFirst 5 rows:\n", df.head())

# ------------------------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------------------------
print("\nStatistical Summary:\n", df.describe())

print("\nCorrelation Matrix:\n", df.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot (important variables only)
sns.pairplot(df[["TV", "Radio", "Newspaper", "Sales"]])
plt.show()

# ------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Feature Scaling 
# ------------------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ------------------------------------------------------------
# Actual vs Predicted Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ------------------------------------------------------------
# Feature Importance (Coefficients)
# ------------------------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n" + "=" * 60)
print("ADVERTISING IMPACT")
print("=" * 60)
print(importance)

plt.figure(figsize=(8, 5))
plt.bar(importance["Feature"], importance["Coefficient"])
plt.title("Advertising Impact on Sales")
plt.xlabel("Channel")
plt.ylabel("Coefficient")
plt.show()

# ------------------------------------------------------------
# Future Sales Forecasting
# ------------------------------------------------------------
future_campaigns = pd.DataFrame({
    "TV": [50, 100, 200, 300],
    "Radio": [10, 20, 40, 60],
    "Newspaper": [5, 10, 20, 30]
})

future_scaled = scaler.transform(future_campaigns)
future_campaigns["Predicted Sales"] = model.predict(future_scaled)

print("\n" + "=" * 60)
print("FUTURE SALES FORECAST")
print("=" * 60)
print(future_campaigns)

# ------------------------------------------------------------
# Custom Campaign Prediction
# ------------------------------------------------------------
custom_campaign = pd.DataFrame({
    "TV": [250],
    "Radio": [50],
    "Newspaper": [30]
})

custom_scaled = scaler.transform(custom_campaign)
predicted_sales = model.predict(custom_scaled)

print("\nPredicted Sales for Custom Campaign:")
print(round(predicted_sales[0], 2))

# ------------------------------------------------------------
# Business Insights
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

best_channel = importance.iloc[0]["Feature"]

print(f"\n1. Most Influential Factor: {best_channel}")
print("2. TV and Radio have strongest impact on sales.")
print("3. Newspaper has relatively lower contribution.")
print("4. Invest more in high-impact channels (TV/Radio).")
print("5. Use model before launching campaigns for prediction.")
print("6. Continuously update model with new data.")
print("7. Optimize budget allocation using coefficients.")

print("\nProject Completed Successfully 🚀")