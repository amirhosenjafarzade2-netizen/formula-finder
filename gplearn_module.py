# gplearn_module.py
"""
Module for GPlearn integration in Formula Discovery App.
Install with: pip install gplearn
Note: Ensure scikit-learn >= 0.23 for compatibility. If '_validate_data' error persists, update sklearn.
"""

from typing import List, Dict, Any
import sympy as sp
from sklearn.metrics import r2_score
from gplearn.genetic import SymbolicRegressor
import numpy as np
import pandas as pd

def discover_gplearn(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_iterations: int = 100,
    max_complexity: int = 10,
    target_name: str = "y"
) -> Dict[str, Any]:
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)
    
    # GPlearn params tuned for brevity and accuracy
    model = SymbolicRegressor(
        population_size=5000,
        generations=n_iterations // 5,  # Fewer generations, larger pop for speed
        parsimony_coefficient=0.01,  # Fixed low value for shorter formulas
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', 'exp'],
        metric='mean squared error',
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=0,
        random_state=42
    )
    model.fit(X_arr, y_arr)
    
    y_pred = model.predict(X_arr)
    score = r2_score(y_arr, y_pred)
    
    # Extract best program as string (Lisp format)
    equation_str = str(model)
    # Replace X0, X1, etc. with feature names
    for i, name in enumerate(feature_names):
        equation_str = equation_str.replace(f'X{i}', name)
    
    # Attempt to parse to SymPy (may fail for complex Lisp; fallback to 0)
    try:
        equation = sp.sympify(equation_str)
    except Exception:
        equation = sp.Float(0)  # Placeholder; evaluation uses model.predict
    
    complexity = len(equation_str.split())  # Rough estimate based on string
    
    str_formula = equation_str  # Keep Lisp as plain text for now
    
    return {
        "equation": equation,
        "str_formula": str_formula,
        "score": float(score),
        "complexity": int(complexity),
        "feature_names": feature_names,
        "target_name": target_name,
        "is_linear": False,
        "method": "GPlearn (Genetic Programming)",
        "model": model  # For numerical evaluation
    }
