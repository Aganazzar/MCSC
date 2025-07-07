# Root Finding Methods Web App

This project is a Python web application built with **Streamlit** that lets users find roots of functions using methods like Bisection, Newton-Raphson, Secant, and Fixed-Point.
---
##  Prerequisites

- Python 3.10 or higher installed on your system
---

## ⚙️ Setup Instructions

1. **Clone or download the project folder**
2. pip install -r requirements.txt
3. MOST IMPORTANT ARE
streamlit
numpy
pandas
matplotlib
sympy
4. streamlit run main.py
To run the code
if did not work try 
 http://localhost:8501


To add a new root-finding method:
Create a new Python file inside methods
Implement the method following the existing pattern
Import it in main.py
Add UI options for it in main.py
Functions should be written in Python math syntax, e.g., x**3 - x - 2
The app will display the math function in readable format automatically
