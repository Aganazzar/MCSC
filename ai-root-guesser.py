import joblib
import sympy as sp
import numpy as np

x = sp.symbols('x')
sample_points = np.linspace(-3, 3, 10)

def extract_features(expr):
    return [float(expr.evalf(subs={x: val})) for val in sample_points]

# Load model + scaler
model = joblib.load("AIrootguesser.pkl")
scaler = joblib.load("AIrootguesserscaler.pkl")

# Example prediction
expr = sp.sin(x) - 0.6
features = extract_features(expr)
features_scaled = scaler.transform([features])
guess = model.predict(features_scaled)[0]

print("🤖 AI Root Guess:", guess)