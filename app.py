# formula_app.py
"""
Standalone Streamlit App for Formula Discovery.
Modernized with PySR (if Julia available), gplearn fallback, and linear backup.
Run with: streamlit run formula_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
from typing import List, Dict, Any
import sympy as sp
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Fix Julia env (for Streamlit Cloud)
os.environ['JULIA_DEPOT_PATH'] = '/tmp/julia'
os.environ['JULIA_LOAD_PATH'] = '/tmp/julia'

# === Backend detection ===
BACKEND = None
try:
    from pysr import PySRRegressor
    BACKEND = "pysr"
except Exception:
    try:
        from gplearn.genetic import SymbolicRegressor
        BACKEND = "gplearn"
    except Exception:
        BACKEND = "linear"


class FormulaDiscoveryError(Exception):
    pass


def discover_formula(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    max_complexity: int = 10,
    n_iterations: int = 100,
    target_name: str = "y",
    use_symbolic: bool = True
) -> Dict[str, Any]:
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    if len(X_arr) == 0 or np.any(np.isnan(X_arr)) or np.any(np.isnan(y_arr)):
        raise FormulaDiscoveryError("Invalid data: NaNs or empty.")

    # === PySR ===
    if use_symbolic and BACKEND == "pysr":
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
                "is_linear": False
            }
        except Exception as e:
            st.warning(f"⚠️ PySR failed ({e}), switching fallback…")

    # === gplearn fallback ===
    if use_symbolic and BACKEND == "gplearn":
        try:
            model = SymbolicRegressor(
                generations=int(n_iterations / 10),
                population_size=1000,
                function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt', 'log'],
                parsimony_coefficient=0.001,
                max_samples=1.0,
                verbose=0,
                random_state=42
            )
            model.fit(X_arr, y_arr)
            y_pred = model.predict(X_arr)
            score = r2_score(y_arr, y_pred)

            str_formula = model._program.__str__()
            # Replace X0, X1, … with actual feature names
            for i, name in enumerate(feature_names):
                str_formula = str_formula.replace(f"X{i}", name)
            try:
                equation = sp.sympify(str_formula)
            except Exception:
                equation = sp.Symbol(str_formula)

            return {
                "equation": equation,
                "str_formula": str_formula,
                "score": float(score),
                "complexity": len(str_formula),
                "feature_names": feature_names,
                "target_name": target_name,
                "is_linear": False
            }
        except Exception as e:
            st.warning(f"⚠️ gplearn failed ({e}), switching to linear…")

    # === Linear fallback ===
    model = LinearRegression()
    model.fit(X_arr, y_arr)
    y_pred = model.predict(X_arr)
    score = r2_score(y_arr, y_pred)

    coeffs = model.coef_
    intercept = model.intercept_
    terms = [sp.Float(coeff) * sp.Symbol(name) for name, coeff in zip(feature_names, coeffs)]
    equation = sum(terms) + sp.Float(intercept)

    return {
        "equation": equation,
        "str_formula": str(equation),
        "score": float(score),
        "complexity": len(feature_names) + 1,
        "feature_names": feature_names,
        "target_name": target_name,
        "is_linear": True
    }


def load_and_preprocess_data(uploaded_files, n_rows=None):
    """Load numeric Excel data or generate sample."""
    if not uploaded_files:
        st.info("Using sample data.")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'Feature1': rng.normal(1.2, 0.05, 100),
            'Feature2': rng.normal(500, 50, 100),
            'Feature3': rng.normal(30, 2, 100),
            'Target': 2 * rng.normal(1.2, 0.05, 100) + np.sin(rng.normal(30, 2, 100))
        })
        return df

    dfs = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
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


# === Streamlit UI ===
st.set_page_config(page_title="Formula Discovery App", layout="wide")

st.title("🧮 Standalone Formula Discovery App")

st.sidebar.header("⚙️ Config")
n_iterations = st.sidebar.number_input("Iterations", min_value=10, value=100, help="Number of search iterations")
max_complexity = st.sidebar.number_input("Max Complexity", min_value=1, value=10, help="Max equation size")
min_rows = st.sidebar.number_input("Min Rows", min_value=5, value=10, help="Minimum data points required")

uploaded_files = st.file_uploader("📁 Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'])
n_rows_input = st.number_input("Sample rows (0 for all)", min_value=0, value=0)
n_rows = None if n_rows_input == 0 else n_rows_input

if st.button("Load Data"):
    df = load_and_preprocess_data(uploaded_files, n_rows)
    if not df.empty:
        st.session_state.df = df
        st.success(f"✅ Loaded {len(df)} rows with {len(df.columns)} numeric columns")
        with st.expander("👁️ Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)

if 'df' not in st.session_state:
    st.warning("⚠️ Load data first.")
    st.stop()

df = st.session_state.df
params = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    formula_features = st.multiselect("🔧 Select Features", options=params, default=params[:-1] if len(params) > 1 else [])
with col2:
    formula_target = st.selectbox("🎯 Target Variable", options=params)

if not formula_features or formula_target not in params or formula_target in formula_features:
    st.error("❌ Select valid features (excluding target).")
    st.stop()

# Method choice
method_choice = st.radio(
    "📊 Method",
    options=["Best Available (Symbolic if possible)", "Linear Only"],
    index=0 if BACKEND != "linear" else 1,
    help="Symbolic: nonlinear discovery (PySR/gplearn). Linear: simple regression."
)
use_symbolic = (method_choice == "Best Available (Symbolic if possible)")

if BACKEND == "linear" and use_symbolic:
    st.warning("⚠️ No symbolic backend available, using linear only.")
    use_symbolic = False

run_formula = st.button("🚀 Discover Formula", type="primary")

if run_formula:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📊 Preparing data...")
        progress_bar.progress(0.2)

        X_formula = df[formula_features].copy()
        y_formula = df[formula_target].copy()

        mask = ~(X_formula.isna().any(axis=1) | y_formula.isna())
        X_formula = X_formula[mask]
        y_formula = y_formula[mask]

        if len(X_formula) < min_rows:
            raise FormulaDiscoveryError(f"Insufficient valid data: {len(X_formula)} rows (need ≥{min_rows})")

        progress_bar.progress(0.4)
        status_text.text("🔍 Running discovery...")

        formula_result = discover_formula(
            X_formula, y_formula, formula_features,
            max_complexity=max_complexity,
            n_iterations=n_iterations,
            target_name=formula_target,
            use_symbolic=use_symbolic
        )

        progress_bar.progress(0.8)
        status_text.text("📈 Generating visualization...")

        fallback_msg = " (Linear Approximation)" if formula_result.get("is_linear") else ""
        st.success(f"✅ Formula discovered{fallback_msg}!")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("📊 R² Score", f"{formula_result['score']:.4f}")
        with col_res2:
            st.metric("🔢 Complexity", formula_result['complexity'])

        st.subheader("📜 Discovered Formula")
        formula_type = "Linear Model" if formula_result.get("is_linear") else "Symbolic Expression"
        st.info(f"**Type:** {formula_type}")

        latex_formula = sp.latex(formula_result['equation'])
        st.latex(latex_formula)

        with st.expander("🔤 Plain Text Version"):
            st.code(formula_result['str_formula'], language='text')

        with st.expander("ℹ️ Details"):
            st.write(f"**Target:** {formula_result['target_name']}")
            st.write(f"**Features Used:** {', '.join(formula_result['feature_names'])}")

        equation = formula_result['equation']
        y_pred = []
        for idx in range(len(X_formula)):
            row = X_formula.iloc[idx]
            val_dict = {sp.Symbol(name): float(row[name]) for name in formula_features}
            try:
                pred_val = float(equation.subs(val_dict).evalf())
                y_pred.append(pred_val)
            except Exception:
                y_pred.append(np.nan)

        y_pred = np.array(y_pred)
        mask_valid = ~np.isnan(y_pred)
        if mask_valid.sum() > 0:
            y_actual_valid = y_formula.values[mask_valid]
            y_pred_valid = y_pred[mask_valid]

            fig = px.scatter(
                x=y_actual_valid, y=y_pred_valid,
                labels={'x': f'Actual {formula_target}', 'y': f'Predicted {formula_target}'},
                title=f'Predictions vs. Actual Values ({formula_type})',
            )
            min_val = min(y_actual_valid.min(), y_pred_valid.min())
            max_val = max(y_actual_valid.max(), y_pred_valid.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Fit',
                line=dict(dash='dash', color='red', width=2)
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Could not evaluate formula on data points.")

        formula_text = f"""Formula Discovery Results
===================

Target: {formula_result['target_name']}
Type: {formula_type}
R² Score: {formula_result['score']:.4f}
Complexity: {formula_result['complexity']}
Features: {', '.join(formula_result['feature_names'])}

LaTeX Formula:
{latex_formula}

Plain Text:
{formula_result['str_formula']}"""
        st.download_button(
            "💾 Download Report", 
            formula_text, 
            f"formula_report_{formula_target}.txt", 
            "text/plain"
        )

        progress_bar.progress(1.0)
    except FormulaDiscoveryError as e:
        st.error(f"❌ Discovery Error: {str(e)}")
    except Exception as e:
        st.error(f"💥 Unexpected Error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()
