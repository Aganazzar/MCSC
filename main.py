# main.py
import sympy as sp
import streamlit as st
import math
from methods.bisection import bisection
from methods.newton import newton
from methods.secant import secant
from methods.fixed_point import fixed_point

st.set_page_config(page_title="Root Solver", layout="centered")

st.title("Root Finding Methods")
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
    if method == "Bisection":
        result, iteration_data, fig = bisection(func_input, a, b, tol, max_iter)

        if result is None:
            st.error("❌ Bisection Method failed: f(a) and f(b) must have opposite signs.")
        else:
            st.success(f"✅ Result: Root ≈ {result}")

            with st.expander("Show More (Explanation, Iterations & Graph)", expanded=False):
                st.markdown("### Method Explanation")
                st.write("""
                The Bisection Method works by repeatedly narrowing the interval [a, b] where the root lies. 
                It checks the midpoint and replaces either a or b based on the sign of f(x).
                It’s guaranteed to work if f(a) and f(b) have opposite signs.
                """)

                st.markdown("###  Iteration Table")
                st.dataframe(iteration_data)

                st.markdown("###  Function Graph")
                st.pyplot(fig)

    elif method == "Fixed-Point":
        result, iteration_data, fig = fixed_point(g_input, x0, tol, max_iter)

        if result is None:
            st.error("❌ Fixed-Point Method failed: Invalid g(x) or no convergence.")
        else:
            st.success(f"✅ Result: Root ≈ {result}")

            with st.expander(" Show More (Explanation, Iterations & Graph)", expanded=False):
                st.markdown("###  Method Explanation")
                st.write("""
                The Fixed-Point Iteration rewrites f(x) = 0 into x = g(x). 
                Then, it repeatedly evaluates xₙ₊₁ = g(xₙ) until convergence.
                Convergence depends on g(x) being well-behaved (Lipschitz condition).
                """)

                st.markdown("###  Iteration Table")
                st.dataframe(iteration_data)

                st.markdown("###  Function Graph")
                st.pyplot(fig)

    elif method == "Newton-Raphson":
        result, iteration_data, fig = newton(func_input, df_input, x0, tol, max_iter)

        if result is None:
            st.error("❌ Newton-Raphson Method failed: Invalid function or derivative.")
        else:
            st.success(f"✅ Result: Root ≈ {result}")

            with st.expander(" Show More (Explanation, Iterations & Graph)", expanded=False):
                st.markdown("###  Method Explanation")
                st.write("""
                Newton-Raphson uses the derivative f'(x) to find better root estimates by:
                xₙ₊₁ = xₙ - f(xₙ) / f'(xₙ).
                It converges fast but requires a good initial guess and differentiable f.
                """)

                st.markdown("###  Iteration Table")
                st.dataframe(iteration_data)

                st.markdown("###  Function Graph")
                st.pyplot(fig)

    elif method == "Secant":
        result, iteration_data, fig = secant(func_input, x0, x1, tol, max_iter)

        if result is None:
            st.error("❌ Secant Method failed: Invalid initial guesses or function.")
        else:
            st.success(f"✅ Result: Root ≈ {result}")

            with st.expander(" Show More (Explanation, Iterations & Graph)", expanded=False):
                st.markdown("###  Method Explanation")
                st.write("""
                The Secant Method approximates the derivative by two points and updates root estimates using:
                xₙ₊₁ = xₙ - f(xₙ) * (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁)).
                It needs two initial guesses and does not require explicit derivatives.
                """)

                st.markdown("###  Iteration Table")
                st.dataframe(iteration_data)

                st.markdown("###  Function Graph")
                st.pyplot(fig)
