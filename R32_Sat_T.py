import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv('R32_P_T.csv')  # CSV should have columns: 'Pressure', 'SatTemp'

# Features and Target
X = df[['Pressure']]
y = df['SatTemp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the Pressure data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Transformation
degree = 4
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train Model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="R32 Saturation Temp Predictor", layout="centered")
st.title("🌡️ R32 Saturation Temperature Predictor")
st.markdown("Enter a pressure (in **PSIG**) to predict the saturation temperature.")

# Input
input_pressure = st.number_input("Enter Pressure (PSIG)", min_value=0.0, max_value=700.0, step=1.0)

if input_pressure:
    # Transform and predict
    input_scaled = scaler.transform(np.array([[input_pressure]]))
    input_poly = poly.transform(input_scaled)
    predicted_temp = model.predict(input_poly)[0]
    st.success(f"🌡️ Predicted Saturation Temp at {input_pressure:.1f} PSIG: **{predicted_temp:.2f} °C**")

# Show Evaluation
st.subheader("📈 Model Performance")
st.write(f"**Degree:** {degree}")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error:** {mse:.4f}")

# Plot the fitted curve
st.subheader("🔍 Curve Fit Visualization")

X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)
X_range_poly = poly.transform(X_range_scaled)
y_range_pred = model.predict(X_range_poly)

fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X_range, y_range_pred, color='red', label='Polynomial Fit')
ax.set_xlabel("Pressure (PSIG)")
ax.set_ylabel("Saturation Temperature (°C)_
