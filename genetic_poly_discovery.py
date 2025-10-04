import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sympy as sp
from pysr import PySRRegressor
from typing import List, Dict, Any

class FormulaDiscoveryError(Exception):
    pass

def discover_genetic_poly(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    max_complexity: int = 10,
    n_iterations: int = 100,
    target_name: str = "y"
) -> Dict[str, Any]:
    """
    Discover a polynomial formula using PySR's genetic algorithm, restricted to polynomial forms.
    """
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    if len(X_arr) == 0 or np.any(np.isnan(X_arr)) or np.any(np.isnan(y_arr)):
        raise FormulaDiscoveryError("Invalid data: NaNs or empty.")

    try:
        model = PySRRegressor(
            niterations=n_iterations,
            binary_operators=["add", "sub", "mul"],
            unary_operators=[],
            maxsize=max_complexity,
            loss="(x, y) -> (x - y)^2",
            model_selection="best",
            verbosity=0,
            progress=False
        )
        model.fit(X_arr, y_arr, variable_names=feature_names)

        y_pred = model.predict(X_arr)
        score = r2_score(y_arr, y_pred)

        equation_str = str(model.sympy())
        equation = sp.sympify(equation_str)
        complexity = len(list(sp.preorder_traversal(equation)))

        str_formula = str(sp.simplify(equation))
        for i, name in enumerate(feature_names):
            str_formula = str_formula.replace(f"x{i}", name)

        return {
            "equation": equation,
            "str_formula": str_formula,
            "score": float(score),
            "complexity": int(complexity),
            "feature_names": feature_names,
            "target_name": target_name,
            "is_linear": False,
            "method": "Genetic Algorithm (Polynomial)"
        }
    except Exception as e:
        raise FormulaDiscoveryError(f"Genetic Polynomial failed: {e}")
