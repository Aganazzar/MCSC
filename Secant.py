# Secant Method

def secant(x0, x1, tol=1e-6, max_iter=100):
    for n in range(1, max_iter + 1):
        f0 = f(x0)
        f1 = f(x1)
        if f1 - f0 == 0:
            raise ZeroDivisionError("Zero denominator in secant formula.")
        
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if abs(x2 - x1) < tol:
            return x2, n

        x0, x1 = x1, x2

    raise ValueError("Exceeded maximum iterations")

# Example usage
x0, x1 = 0.5, 1.0
root, iterations = secant(x0, x1)
print(f"Secant Root: {root}")
print(f"Iterations: {iterations}")
