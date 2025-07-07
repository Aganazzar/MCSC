import tkinter as tk
from tkinter import ttk, messagebox
from sympy import symbols, sympify, lambdify, diff, solve,Eq, simplify
import sympy as sp
import numpy as np
import math
import joblib

x=symbols('x')

# to detect zero div errors:
def check_definition(f, x):
    try:
        val = f(x)
        if not np.isfinite(val):  # catches NaN, inf
            return None
        return val
    except Exception:
        return None

# to detect undefined function values at points:
def shift_interval(f, x0, x1):
    max_attempts= 50
    min_shift= 1e-3
    max_shift= 0.05
    shift= min(abs(x1-x0)*0.01, max_shift)
    shift= max(shift, min_shift)

    attempts=0
    while attempts < max_attempts:
        f0 = check_definition(f, x0)
        f1 = check_definition(f, x1)

        if f0 is not None and f1 is not None:
            return x0, x1  # ✅ safe values

        if f0 is None:
            x0 += shift if x0 < x1 else -shift
        if f1 is None:
            x1 -= shift if x0 < x1 else -shift

        attempts += 1

    raise ValueError("Unable to shift interval to avoid undefined function values.")


# parser in budget:
import re

def preprocess_expression(expr):
    # Replace ^ with ** for exponentiation
    expr = expr.replace("^", "**")

    # Fix implicit multiplication (e.g. 2x → 2*x, xsin(x) → x*sin(x))
    expr = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", expr)  # 2x or 3sin(x) → 2*x or 3*sin(x)
    expr = re.sub(r"([a-zA-Z\)])(\()", r"\1*\2", expr)  # x( or sin(x)( → x*( or sin(x)*(
    
    # Insert * between variable and function, e.g. xsin -> x*sin
    functions = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log", "exp", "sqrt"]
    pattern = r"([a-zA-Z0-9])(" + "|".join(functions) + r")"
    expr = re.sub(pattern, r"\1*\2", expr)


    expr = re.sub(
        r"sin\s*\^\s*-1\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?",
        r"asin(\1)",
        expr,
    )
    expr = re.sub(
        r"cos\s*\^\s*-1\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?",
        r"acos(\1)",
        expr,
    )
    expr = re.sub(
        r"tan\s*\^\s*-1\s*\(?\s*([a-zA-Z0-9_]+)\s*\)?",
        r"atan(\1)",
        expr,
    )
    
    # Replace sinx with sin(x), cosx with cos(x), etc.
    expr = re.sub(
        r"(?<![a-zA-Z0-9_])(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|asinh|acosh|atanh|log|exp|sqrt)\s*x",
        r"\1(x)",
        expr,
    )
    return expr

# thrid degree Lagrange interpolation and its root
def lagrange_interpolate(f):
    x_vals = np.linspace(-2, 2, 4)  # → 4 points
    y_vals = f(x_vals)

    P=0
    for i in range(4):
        xi = x_vals[i]
        yi = y_vals[i]
        Li = 1
        for j in range(4):
            if i != j:
                xj = x_vals[j]
                Li *= (x - xj) / (xi - xj)
        P += yi * Li

    P= simplify(P)
    print(f"[LAGRANGE] Interpolated polynomial: {P}")

    roots= solve(Eq(P,0),x)
    #print(f"[LAGRANGE] Roots of interpolated polynomial: {roots}")
    return roots


# AI root guesser:

# Load model + scaler for ai guess:
model = joblib.load("AIrootguesser.pkl")
scaler = joblib.load("AIrootguesserscaler.pkl")

def AI(f, function):
    try:
        print("[AI] AI function started", flush=True)
        sample_points = np.linspace(-3, 3, 10)
        def extract_features():
            values = []
            for val in sample_points:
                try:
                    result = f(val)
                    if np.isfinite(result):
                        values.append(float(result))
                    else:
                        values.append(0.0)
                except Exception:
                    values.append(0.0)
            return values

        features = extract_features()
        print(f"[AI] Features extracted: {features}", flush=True)

        features_scaled = scaler.transform([features])
        guess = model.predict(features_scaled)[0]
        print(f"[AI] Predicted root guess: {guess}", flush=True)
        return guess
    except Exception as e:
        print(f"[AI] Error in AI function: {e}", flush=True)
        # fallback guess:
        return 0.0
    
#detect the presence of exponential functions in f:
def contains_exponential(expr):
    for subexpr in sp.preorder_traversal(expr):
        if isinstance(subexpr, sp.exp):
            return True
        if isinstance(subexpr, sp.Pow):
            if subexpr.base == sp.E:  # e^x form
                return True
            if isinstance(subexpr.base, sp.Symbol) and str(subexpr.base) == 'e':
                return True
    return False


    
def is_effectively_real(r):
    # Check if imaginary part is smaller than tol
    tol= 1e-12
    _, imag = sp.sympify(r).as_real_imag()
    return abs(imag.evalf()) < tol

def root_select(f, f_prime, function):
    print("\n--- Root Selection Started ---", flush= True)

    # skip interpolation if exponenetial is detected
    if contains_exponential(function):
        print("Exponential detected.")
        ai= AI(f, function)
        ai_guess= float(ai)
        return ai_guess
    
    roots= lagrange_interpolate(f)
    print(f"Lagrange roots: {roots}")
    ai= AI(f, function)
    ai_guess=float(ai)
    print(f"[AI] Predicted root guess: {ai_guess}")
    # derivative of the interpolating polynomial

    # filter roots:
    print(f"[DEBUG] Starting FILTER loop over {len(roots)} roots...", flush=True)

    safe_roots= []
    low_slope_roots= []

    for r in roots:
        try:
            
            if is_effectively_real(r):
                slope = abs(f_prime.subs(x, r).evalf())
                print(f"[FILTER] Interpolated root: {r}, slope: {slope}", flush=True)
                if slope > 1e-8:
                    safe_roots.append((float(sp.re(r)), slope))
                else:
                    low_slope_roots.append((float(sp.re(r)), slope))
                    print(f"[FILTER] Root {r} discarded due to low slope.", flush=True)
        except Exception as e:
            print(f"[ERROR] Exception while processing root {r}: {e}", flush=True)

    # append AI guess too:
    print(f"[FILTER] AI guess value: {ai_guess}", flush=True)

    try:
        ai_slope = abs(f_prime.subs(x, ai_guess).evalf())
        print(f"[FILTER] AI guess slope: {ai_slope}", flush=True)
    except Exception as e:
        print(f"[FILTER] Exception computing AI slope: {e}", flush=True)

    if ai_slope> 1e-8:
        safe_roots.append((float(ai_guess), ai_slope))
        print(f"[FILTER] AI guess appended.")

    else: 
        low_slope_roots.append((float(ai_guess), ai_slope))
        print(f"[FILTER] AI guess discarded due to low slope.")

    if safe_roots:
        # Use steepest root among candidates
        best_root= max(safe_roots, key=lambda tup: tup[1])[0]
        print(f"[SELECT] Best root chosen (steepest): {best_root}")
        return float(best_root)
    

    # Otherwise fallback to the root with the highest slope among low slopes
    if low_slope_roots:
        fallback_root = max(low_slope_roots, key=lambda tup: tup[1])[0]
        print(f"[FALLBACK] No steep slope root found. Using best low slope root: {fallback_root}", flush=True)
        return float(fallback_root)

    
    # if nothing works,
    print("[FALLBACK] No good interpolation root. Using AI fallback.")
    return float(ai_guess)

# iterative method(newton-raphson)
def iterate(f, f_prime ,x0, tol, max_iter):
    print(f"Iterative method started: {x0}")
    xn= x0
    fxn=f(xn)
    diverge_counter= 0

    print(f"diverge counter: {fxn}")
    
    if abs(xn) < 1/tol and abs(fxn) < tol:
        print(f"hey: {xn}")
        return float(xn), 0

    print("abs")
    prev_fx= abs(fxn)
    print("prev_fx")
   
    for n in range(1, max_iter+1):
        print("Strating interations")
        fx= f(xn)

        fpx= f_prime(xn)
        if abs(fpx)< 1e-12:
            print(f"Derivative value is zero")
            raise ValueError("Error")
        
        x_next= xn-fx/fpx
        fx_next= abs(f(x_next))

        # divergence check:
        if fx_next>prev_fx:
            diverge_counter+=1
        else:
            diverge_counter=0

        if diverge_counter>=3:
            print(f"Divergence: {diverge_counter}")
            raise ValueError("Fucntion diverging after {n} iterations.")

        if abs(x_next-xn)<tol and abs(f(x_next))< tol:
            return x_next, n
        
        prev_fx= fx_next
        xn= x_next 

    # maximum iterations exceeded1
    raise ValueError("No real root found within the interval.")

# bracketing algo:
def bracket(f, f_prime ,x0, x1, tol, max_iter):
     # Ensure the root is bracketed


    try:
        x0, x1= shift_interval(f, x0, x1)
    except Exception as e:
        raise ValueError("Shifting Error")
    
    if f(x0) * f(x1) >= 0:
        try:
            try:
                root,n= iterate(f, f_prime ,(x0+x1)/2, tol, max_iter)
            except Exception as e:
                raise ValueError("NR Error")
            if (min(x0, x1) < root < max(x0, x1)):
                return root,n
            else:
               raise ValueError("Root lies outside interval")
        except Exception as e:
            raise ValueError("Root does not exist within interval")

    for n in range(1, max_iter+1):

        try:
            x0, x1= shift_interval(f, x0, x1)
        except Exception as e:
            raise ValueError("Shifitng Error")
        
        f0=f(x0)
        f1=f(x1)

        # Start with Secant Method
        if abs(f1-f0)> 1e-12:
            x2= x1-f1*(x1-x0)/(f1-f0)
        else:
            x2=(x0+x1)/2
 
        f2=check_definition(f,x2)
        if f2 is None:
            # Try shifting left/right to find a nearby defined value
            try:
                x0, x2 = shift_interval(f, x0, x2)
                f2 = f(x2)
            except:
                raise ValueError("Shifting Error")

        if f0!=f2 and f1!=f2 and f0!=f1:
            try:
            #IQI formula, so that the denominator!=0
                x2= (x0*f1*f2)/((f0-f1)*(f0-f2))+(x1*f0*f2)/((f1-f0)*(f1-f2))+x2*(f0*f1)/((f2-f0)*(f2-f1))
                f2=f(x2)
            except ZeroDivisionError:
                pass # skip IQI

        # Safety checks: if s is not between x1 and x2 or not making progress

        if not (min(x0, x1) < x2 < max(x0, x1)) or abs(x2 - x1) < tol:
            x2 = (x0+x1)/2 # Fall bxack to Bisection
            f2=check_definition(f,x2)
            if f2 is None:
                try:
                    x0, x2 = shift_interval(f, x0, x2)
                    f2 = f(x2)
                except:
                    raise ValueError("Shifting Error")
            
        #update the bracket
        if f0*f2<0:
            x1=x2
        else:
            x0=x2

        #Convergence check
        if abs(x1-x0)<tol or abs(f2)<tol:
            if not abs(f2)>tol:
                return x2,n

    # Failed to converge
    raise ValueError("No real root found within the interval.")
 
 # GUI setup
root=tk.Tk()
root.title("Find x: Root Finder")

def update_input_fields(event=None):
    method = method_choice.get()
    if method == "Enter the interval within which to find the root":
        # Show both interval entries
        label_x0.grid()
        x0_entry.grid()
        label_x1.grid()
        x1_entry.grid()
        label_guess.grid_remove()
        guess_entry.grid_remove()
    else:
        # Show only initial guess entry
        label_guess.grid()
        guess_entry.grid()
        label_x0.grid_remove()
        x0_entry.grid_remove()
        label_x1.grid_remove()
        x1_entry.grid_remove()

def calculate():
    try:
        expression= preprocess_expression(function_entry.get())
        method= method_choice.get()
        function=sympify(expression, locals={'e': sp.E})
        fp= diff(function,x)
        f= lambdify(x, function, modules=["numpy"])
        f_prime= lambdify(x, fp, modules=["numpy"])
        
        tol_str= error_tolerance.get().strip()
        tol= float(sympify(tol_str)) if tol_str else 1e-6

        root_value= None

        if method == "Enter the interval within which to find the root":
            x0_string= x0_entry.get()
            x1_string= x1_entry.get()

            if not x0_string or not x1_string:
                raise ValueError("Please enter both the endpoints.")

            x0= float(x0_string)
            x1=float(x1_string) 

            if x1 is None:
               raise ValueError("Please provide both x0 and x1")
            root_value, iterations= bracket(f,f_prime ,x0, x1, tol, 500)

        elif method == "Just find any real root for the function":
            x0_entry_string= root_select(f, fp, function)
            x0= float(x0_entry_string)

            # Auto-fill the guess_entry field with the selected x0
            guess_entry.delete(0, tk.END)
            guess_entry.insert(0, str(round(x0)))            
            root_value, iterations= iterate(f, f_prime ,x0, tol, 500)

        result_label.config(text=f"Result: {root_value} (found in {iterations} iterations)")
    except Exception as e:
        messagebox.showerror("Error", f"Error:\n{e}")


# Widgets
tk.Label(root, text="Enter function f(x):").grid(row=0, column=0)
function_entry= tk.Entry(root, width=50)
function_entry.grid(row=0, column=1)

tk.Label(root, text="Choose a method:").grid(row=1, column=0)
method_choice= ttk.Combobox(root, values=["Just find any real root for the function", "Enter the interval within which to find the root"], state="readonly")
method_choice.grid(row=1, column=1)
method_choice.current(0)
method_choice.bind("<<ComboboxSelected>>", update_input_fields)

# Interval inputs
label_x0 = tk.Label(root, text="x0:")
x0_entry = tk.Entry(root)
label_x1 = tk.Label(root, text="x1:")
x1_entry = tk.Entry(root)

label_x0.grid(row=2, column=0)
x0_entry.grid(row=2, column=1)
label_x1.grid(row=2, column=2)
x1_entry.grid(row=2, column=3)

# Initial guess input
label_guess = tk.Label(root, text="Initial guess x0:")
guess_entry = tk.Entry(root, state="readonly")
label_guess.grid(row=2, column=0)
guess_entry.grid(row=2, column=1)
label_guess.grid_remove()
guess_entry.grid_remove()

result_label = tk.Label(root, text="Result: ")
result_label.grid(row=5, column=0, columnspan=2)

tk.Label(root, text="Error tolerance:").grid(row=4, column=0)
error_tolerance= tk.Entry(root)
error_tolerance.grid(row=4, column= 1)

tk.Button(root, text="Calculate Root", command=calculate).grid(row=3, column=0, columnspan=2, pady=10)


# Initialize correct visibility
update_input_fields()

root.mainloop()