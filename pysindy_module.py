# pysindy_module.py
"""
Module for PySINDy integration in Formula Discovery App.
Install with: pip install pysindy
"""

from typing import List, Dict, Any
import sympy as sp
from sklearn.metrics import r2_score
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
import numpy as np
import pandas as pd

def discover_pysindy(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    poly_degree: int = 2,
    target_name: str = "y"
) -> Dict[str, Any]:
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)
    
    # Build library of candidate functions (polynomials + interactions)
    library = PolynomialLibrary(
        degree=poly_degree,
        include_bias=False,
        interaction_only=False
    )
    
    # SINDy for regression: fit to y as "output"
    model = SINDy(
        feature_library=library,
        optimizer="STLSQ",  # Sequentially Thresholded Least Squares for sparsity
        threshold=0.1,  # Threshold for sparsity (tune lower for more terms)
        n_jobs=1,
        quiet=True
    )
    
    # For static regression y = f(X), we fit library(X) ~ y
    # PySINDy typically for dx/dt, but this approximates regression
    library_features = library.fit_transform(X_arr)
    model.optimizer_.fit(library_features, y_arr)  # Direct fit on library
    
    # Predict via library
    y_pred = model.predict(X_arr)
    score = r2_score(y_arr, y_pred.ravel() if y_pred.ndim > 1 else y_pred)
    
    # Get coefficients and build equation
    coeffs = model.coefficients()
    feature_names_lib = library.get_feature_names(feature_names)
    
    terms = []
    for coef, name in zip(coeffs.flatten(), feature_names_lib):
        if abs(coef) > 1e-3:  # Threshold for inclusion
            term = sp.Float(coef) * sp.sympify(name.replace(" ", "*"))
            terms.append(term)
    
    if terms:
        equation = sum(terms)
    else:
        equation = sp.Float(0)
    
    complexity = len(terms)
    str_formula = str(sp.simplify(equation))
    
    return {
        "equation": equation,
        "str_formula": str_formula,
        "score": float(score),
        "complexity": int(complexity),
        "feature_names": feature_names,
        "target_name": target_name,
        "is_linear": poly_degree == 1,
        "method": f"PySINDy (Sparse ID, Degree {poly_degree})"
    }
