# gplearn_module.py
"""
Module for GPlearn integration in Formula Discovery App.
Install with: pip install gplearn
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
        parsimony_coefficient_difference=lambda x: x * 0.01,  # Adaptive
        random_state=42,
        function_set_arguments={
            'sqrt': {'arity': 1},
            'log': {'arity': 1},
            'sin': {'arity': 1},
            'cos': {'arity': 1},
            'exp': {'arity': 1},
        }
    )
    model.fit(X_arr, y_arr)
    
    y_pred = model.predict(X_arr)
    score = r2_score(y_arr, y_pred)
    
    # Extract best program as string
    best_program = str(model._program[0])
    # Convert GPlearn lisp-like to infix (simplified; assumes basic structure)
    # For full multi-var, replace X0, X1, etc.
    equation_str = best_program.replace('X0', feature_names[0] if len(feature_names) > 0 else 'x')
    for i, name in enumerate(feature_names[1:], 1):
        equation_str = equation_str.replace(f'X{i}', name)
    
    try:
        equation = sp.sympify(equation_str)
    except:
        # Fallback: use a simple linear if parsing fails
        equation = sp.symbols('x')
        equation_str = 'x'  # Placeholder
        score = 0.0
    
    complexity = len(str(equation).split())  # Rough estimate
    
    str_formula = str(sp.simplify(equation))
    
    return {
        "equation": equation,
        "str_formula": str_formula,
        "score": float(score),
        "complexity": int(complexity),
        "feature_names": feature_names,
        "target_name": target_name,
        "is_linear": False,
        "method": "GPlearn (Genetic Programming)"
    }
