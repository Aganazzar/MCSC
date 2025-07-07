# main.py
import sympy as sp
import streamlit as st
import math
from methods.bisection import bisection
from methods.newton import newton
from methods.secant import secant
from methods.fixed_point import fixed_point

st.set_page_config(page_title="Root Solver", layout="centered")

st.title("🔍 Root Finding Methods")
st.markdown("Solve equations of the form **f(x) = 0** using different numerical methods.")

func_input = st.text_input("Enter your function f(x)", "x**3 - x - 2", help="Use Python syntax: x**2 for x², math.sqrt(x) for √x")
st.write("### Your Function (Formatted)")

try:
    x = sp.Symbol('x')
    parsed_expr = sp.sympify(func_input)
    latex_expr = sp.latex(parsed_expr)
    st.latex(f"f(x) = {latex_expr}")
except:
    st.warning("⚠️ Couldn't render the function — check for typos like x^^2 or missing parenthesis.")

method = st.selectbox("Select Method", ["Bisection", "Fixed-Point", "Newton-Raphson", "Secant"])

a = b = x0 = x1 = None
tol = st.number_input("Tolerance (ε)", value=1e-5, format="%.1e")
max_iter = st.slider("Maximum Iterations", 10, 1000, 100)

if method == "Bisection":
    a = st.number_input("Interval start (a)", value=1.0)
    b = st.number_input("Interval end (b)", value=2.0)
elif method == "Fixed-Point":
    x0 = st.number_input("Initial guess x0", value=1.0)
    g_input = st.text_input("Enter g(x)", "math.sqrt(x + 2)")
elif method == "Newton-Raphson":
    x0 = st.number_input("Initial guess x0", value=1.0)
    df_input = st.text_input("Enter f'(x)", "3*x**2 - 1")
elif method == "Secant":
    x0 = st.number_input("First guess x0", value=1.0)
    x1 = st.number_input("Second guess x1", value=2.0)

# Safe eval
def evaluate(fx, x):
    try:
        return eval(fx, {"x": x, "math": math})
    except Exception as e:
        return None

if st.button("Find Root"):
    result = None
    if method == "Bisection":
        result = bisection(func_input, a, b, tol, max_iter)
    elif method == "Fixed-Point":
        result = fixed_point(g_input, x0, tol, max_iter)
    elif method == "Newton-Raphson":
        result = newton(func_input, df_input, x0, tol, max_iter)
    elif method == "Secant":
        result = secant(func_input, x0, x1, tol, max_iter)

    st.success(f"Result: {result}")
