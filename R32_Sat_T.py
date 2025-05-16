import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Title
st.title("🌡️ R32 Saturation Temperature Predictor")
st.markdown("Developed by [Hasan Samir Hasan]")

st.markdown("Enter a pressure (PSIG) to predict the saturation temperature (°C) using polynomial regression.")

# --- 1. Built-in R32 Pressure-Temperature Data
data = {
    "Pressure": [
        11, 14.4, 18.2, 22.3, 26.8, 31.7, 37.1, 42.9, 49.3, 56.1, 63.5, 71.5, 80, 89.2,
        99.1, 109.7, 121, 133, 145.9, 159.5, 174.1, 189.5, 205.8, 223.2, 241.5, 260.9,
        281.3, 302.9, 325.7, 349.6, 374.9, 401.4, 429.3, 458.6, 489.4, 521.8, 555.7, 591.4, 628.8
    ],
    "SatTemp": [
        -40, -37.2, -34.4, -31.7, -28.9, -26.1, -23.3, -20.6, -17.8, -15, -12.2, -9.4, -6.7,
        -3.9, -1.1, 1.7, 4.4, 7.2, 10, 12.8, 15.6, 18.3, 21.1, 23.9, 26.7, 29.4, 32.2, 35,
        37.8, 40.6, 43.3, 46.1, 48.9, 51.7, 54.4, 57.2, 60, 62.8, 65.6
    ]
}
df = pd.DataFrame(data)
# --- 2. Train a Polynomial Regression Model
X = df[['Pressure']]
y = df['SatTemp']
degree = 4  # You can try 3 or 5 if desired
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# --- 3. User Input
pressure_input = st.number_input("Enter Pressure (PSIG):", min_value=0.0, value=630.0, step=0.1)

# --- 4. Predict
input_poly = poly.transform(np.array([[pressure_input]]))
predicted_temp = model.predict(input_poly)[0]
st.success(f"Predicted Saturation Temperature: {predicted_temp:.2f} °C")

# --- 5. Optional: Show Curve
if st.checkbox("📈 Show Curve"):
    import matplotlib.pyplot as plt

    X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_range = model.predict(poly.transform(X_range))

    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data')
    ax.plot(X_range, y_range, color='red', label='Polynomial Fit')
    ax.set_xlabel('Pressure (PSIG)')
    ax.set_ylabel('Saturation Temp (°C)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
