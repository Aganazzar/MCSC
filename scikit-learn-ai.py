import numpy as np
import sympy as sp
from sympy import sin, cos, exp, log
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ----- SETTINGS -----
BATCH_SIZE = 10000
BATCHES = 25  # 25 × 10,000 = 250,000
SAMPLE_POINTS = np.linspace(-3, 3, 10)

x = sp.symbols('x')

# ----- FEATURE EXTRACTION -----
def extract_features(expr):
    return [float(expr.evalf(subs={x: val})) for val in SAMPLE_POINTS]

# ----- MODEL & SCALER INIT -----
model = MLPRegressor(hidden_layer_sizes=(64, 64), warm_start=True, max_iter=1)
scaler = StandardScaler()
model_initialized = False

# ----- BATCHED TRAINING -----
for batch_num in range(BATCHES):
    X_batch = []
    y_batch = []

    with tqdm(total=BATCH_SIZE, desc=f"Batch {batch_num+1}/{BATCHES}") as pbar:
        while len(X_batch) < BATCH_SIZE:
            eq_type = random.choice(["quadratic", "cubic", "sin", "exp", "log"])
            try:
                if eq_type == "quadratic":
                    a, b, c = np.random.uniform(1, 5), np.random.uniform(-10, 10), np.random.uniform(-10, 10)
                    expr = a*x**2 + b*x + c
                elif eq_type == "cubic":
                    a, b, c, d = [np.random.uniform(-5, 5) for _ in range(4)]
                    expr = a*x**3 + b*x**2 + c*x + d
                elif eq_type == "sin":
                    c = np.random.uniform(-1, 1)
                    expr = sin(x) - c
                elif eq_type == "exp":
                    c = np.random.uniform(0.1, 5)
                    expr = exp(x) - c
                elif eq_type == "log":
                    c = np.random.uniform(-5, 5)
                    expr = log(x) - c

                root = sp.nsolve(expr, x, 0)

                if root.is_real and abs(root) < 10:
                    features = extract_features(expr)
                    X_batch.append(features)
                    y_batch.append(float(root))
                    pbar.update(1)
            except Exception:
                continue

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)

    # Scale features
    if not model_initialized:
        scaler.fit(X_batch)
        model_initialized = True

    X_scaled = scaler.transform(X_batch)

    # Fit model incrementally
    model.partial_fit(X_scaled, y_batch)

# ----- SAVE MODEL -----
joblib.dump(model, "AIrootguesser.pkl")
joblib.dump(scaler, "AIrootguesserscaler.pkl")
print("✅ Model and scaler saved.")
