# formula_app.py
"""
Standalone Streamlit App for Formula Discovery.
Isolated from main dashboard. Uses PySR for modern symbolic regression.
Run with: streamlit run formula_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from typing import List, Dict, Any
import sympy as sp
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression  # Fallback

# PySR import with fallback
PYSR_AVAILABLE = False
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except (ImportError, Exception) as e:
    st.warning(f"PySR unavailable ({e}); using linear fallback.")

class FormulaDiscoveryError(Exception):
    pass

def discover_formula(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    max_complexity: int = 10,
    n_iterations: int = 100,
    target_name: str = "y"
) -> Dict[str, Any]:
    """Discover formula using PySR (or linear fallback)."""
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    if len(X_arr) == 0 or np.any(np.isnan(X_arr)) or np.any(np.isnan(y_arr)):
        raise FormulaDiscoveryError("Invalid data: NaNs or empty.")

    if PYSR_AVAILABLE:
        try:
            model = PySRRegressor(
                niterations=n_iterations,
                binary_operators=["add", "sub", "mul", "div"],
                unary_operators=["exp", "log", "sin", "cos", "sqrt"],
                maxsize=max_complexity,
                loss="(x, y) -> (x - y)^2",
                model_selection="best",
                verbosity=0,
                progress=False
            )
            model.fit(X_arr, y_arr, variable_names=feature_names)

            y_pred = model.predict(X_arr)
            score = r2_score(y_arr, y_pred)

            # Extract SymPy equation
            equation_str = model.sympy()
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
                "target_name": target_name
            }
        except Exception as e:
            st.warning(f"PySR failed ({e}); using linear fallback.")
            raise  # Re-raise for now, but fallback below

    # Fallback: Simple linear regression + SymPy expression
    model = LinearRegression()
    model.fit(X_arr, y_arr)
    y_pred = model.predict(X_arr)
    score = r2_score(y_arr, y_pred)

    # Build SymPy linear equation
    coeffs = model.coef_
    intercept = model.intercept_
    terms = [sp.Symbol(name) * coeff for name, coeff in zip(feature_names, coeffs)]
    equation = sum(terms) + intercept
    str_formula = str(sp.simplify(equation))
    complexity = len(feature_names) + 1  # Basic count

    return {
        "equation": equation,
        "str_formula": str_formula,
        "score": float(score),
        "complexity": int(complexity),
        "feature_names": feature_names,
        "target_name": target_name,
        "fallback": True
    }

def load_and_preprocess_data(uploaded_files, n_rows=None):
    """Simple data loader."""
    if not uploaded_files:
        st.info("Using sample data.")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'Feature1': rng.normal(1.2, 0.05, 100),
            'Feature2': rng.normal(500, 50, 100),
            'Feature3': rng.normal(30, 2, 100),
            'Target': 2 * rng.normal(1.2, 0.05, 100) + np.sin(rng.normal(30, 2, 100))  # For testing
        })
        return df

    dfs = []
    for uploaded_file in uploaded_files:
        df_temp = pd.read_excel(io.BytesIO(uploaded_file.read()), engine='openpyxl')
        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df_temp = df_temp[numeric_cols].fillna(df_temp[numeric_cols].median())
            if n_rows:
                df_temp = df_temp.sample(n=min(n_rows, len(df_temp)), random_state=42)
            dfs.append(df_temp)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

st.set_page_config(page_title="Formula Discovery App", layout="wide")

st.title("Standalone Formula Discovery App")

# Sidebar config
st.sidebar.header("Config")
n_iterations = st.sidebar.number_input("Iterations", min_value=10, value=100)
max_complexity = st.sidebar.number_input("Max Complexity", min_value=1, value=10)
min_rows = st.sidebar.number_input("Min Rows", min_value=5, value=10)

# File upload
uploaded_files = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'])
n_rows_input = st.number_input("Sample rows (0 for all)", min_value=0, value=0)
n_rows = None if n_rows_input == 0 else n_rows_input

if st.button("Load Data"):
    df = load_and_preprocess_data(uploaded_files, n_rows)
    if not df.empty:
        st.session_state.df = df
        st.success(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        with st.expander("Preview"):
            st.dataframe(df.head())

if 'df' not in st.session_state:
    st.warning("Load data first.")
    st.stop()

df = st.session_state.df
params = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    formula_features = st.multiselect("Select Features", params, default=params[:-1] if len(params) > 1 else [])
with col2:
    formula_target = st.selectbox("Target", params)

if not formula_features or formula_target not in params:
    st.error("Select features and target.")
    st.stop()

run_formula = st.button("Discover Formula")

if run_formula:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Preparing data...")
        progress_bar.progress(0.2)

        X_formula = df[formula_features].copy()
        y_formula = df[formula_target].copy()

        mask = ~(X_formula.isna().any(axis=1) | y_formula.isna())
        X_formula = X_formula[mask]
        y_formula = y_formula[mask]

        if len(X_formula) < min_rows:
            raise FormulaDiscoveryError(f"Insufficient data: {len(X_formula)} rows")

        progress_bar.progress(0.4)
        status_text.text("Running symbolic regression...")

        formula_result = discover_formula(
            X_formula, y_formula, formula_features,
            max_complexity=max_complexity,
            n_iterations=n_iterations,
            target_name=formula_target
        )

        progress_bar.progress(0.8)
        status_text.text("Generating plot...")

        # Display results
        st.success("Discovery complete!" + (" (Linear fallback used)" if formula_result.get("fallback") else ""))
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("R² Score", f"{formula_result['score']:.4f}")
        with col_res2:
            st.metric("Complexity", formula_result['complexity'])

        st.subheader("Discovered Formula")
        st.code(formula_result['str_formula'])

        # Plot
        equation = formula_result['equation']
        y_pred = []
        for idx in range(len(X_formula)):
            row = X_formula.iloc[idx]
            val_dict = {sp.Symbol(name): float(row[name]) for name in formula_features}
            try:
                pred_val = float(equation.subs(val_dict).evalf())
                y_pred.append(pred_val)
            except:
                y_pred.append(np.nan)

        y_pred = np.array(y_pred)
        mask_valid = ~np.isnan(y_pred)
        if mask_valid.sum() > 0:
            y_actual_valid = y_formula.values[mask_valid]
            y_pred_valid = y_pred[mask_valid]

            fig = px.scatter(
                x=y_actual_valid, y=y_pred_valid,
                labels={'x': f'Actual {formula_target}', 'y': f'Predicted {formula_target}'},
                title='Predictions vs Actual'
            )
            min_val = min(y_actual_valid.min(), y_pred_valid.min())
            max_val = max(y_actual_valid.max(), y_pred_valid.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', name='Perfect Fit',
                                     line=dict(dash='dash', color='red')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not evaluate formula.")

        # Download
        formula_text = f"Formula for {formula_target}\nScore: {formula_result['score']:.4f}\nComplexity: {formula_result['complexity']}\n\n{formula_result['str_formula']}"
        st.download_button("Download Formula", formula_text, f"formula_{formula_target}.txt", "text/plain")

        progress_bar.progress(1.0)
    except FormulaDiscoveryError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()
