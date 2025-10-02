"""
Standalone Streamlit App for Formula Discovery.
Modernized with PySR (if Julia available), PhySO (GA-based symbolic regression), and Linear fallback.
Run with: streamlit run formula_app.py

This version:
- Handles multiple PhySO API versions (get_infix/get_infix_sympy or older/newer variants).
- Falls back to a linear model automatically if PhySO cannot produce a usable sympy expression.
- Tries to avoid julia/torch import ordering issues by importing juliacall early if present.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import sympy as sp
from typing import List, Dict, Any
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# === Environment tweaks / early imports to reduce julia/torch ordering warning ===
# If juliacall is present, import it early to reduce 'torch imported before juliacall' warnings.
# This is optional and will silently continue if juliacall isn't installed.
try:
    import juliacall  # noqa: F401
except Exception:
    pass

# Fix Julia env (for Streamlit Cloud or ephemeral environments)
os.environ.setdefault('JULIA_DEPOT_PATH', '/tmp/julia')
os.environ.setdefault('JULIA_LOAD_PATH', '/tmp/julia')

# === Backend detection ===
pysr_available = False
physo_available = False
try:
    from pysr import PySRRegressor  # type: ignore
    pysr_available = True
except Exception:
    pysr_available = False

try:
    import physo  # type: ignore
    physo_available = True
except Exception:
    physo_available = False

linear_available = True  # Always available

# === Custom exception ===
class FormulaDiscoveryError(Exception):
    pass


def _linear_fallback(X_arr: np.ndarray, y_arr: np.ndarray, feature_names: List[str], target_name: str):
    """
    Fit a simple linear regression and return a dictionary consistent with discover_formula outputs.
    Used as a safe fallback when complex SR backends fail.
    """
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
        "str_formula": str(sp.simplify(equation)),
        "score": float(score),
        "complexity": len(feature_names) + 1,
        "feature_names": feature_names,
        "target_name": target_name,
        "is_linear": True,
        "method": "Linear Regression (Fallback)"
    }


def discover_formula(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    max_complexity: int = 10,
    n_iterations: int = 100,
    target_name: str = "y",
    method: str = "pysr"  # 'pysr', 'physo', 'linear'
) -> Dict[str, Any]:
    """
    Discover a formula mapping features -> target using selected method.
    Returns a dict containing: equation (sympy), str_formula, score, complexity, feature_names, target_name, is_linear, method
    """
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    if len(X_arr) == 0 or np.any(np.isnan(X_arr)) or np.any(np.isnan(y_arr)):
        raise FormulaDiscoveryError("Invalid data: NaNs or empty.")

    # === PySR (Evolutionary Symbolic Regression) ===
    if method == "pysr" and pysr_available:
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

            # model.sympy() returns a sympy.Expression-like object; stringify and sympify to be safe
            equation_str = str(model.sympy())
            equation = sp.sympify(equation_str)
            complexity = len(list(sp.preorder_traversal(equation)))

            str_formula = str(sp.simplify(equation))
            # Replace x0, x1... placeholders if present
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
                "method": "PySR (Evolutionary)"
            }
        except Exception as e:
            raise FormulaDiscoveryError(f"PySR failed: {e}")

    # === PhySO (GA-based Symbolic Regression) ===
    if method == "physo" and physo_available:
        try:
            # Import torch *after* attempting to import juliacall earlier
            import torch  # type: ignore

            # Seed for reproducibility
            seed = 42
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Prepare data as torch tensors
            X_torch = torch.from_numpy(X_arr.T).float()   # PhySO expects shape (n_features, n_samples)
            y_torch = torch.from_numpy(y_arr).float()     # shape (n_samples,)

            # Dummy units (since no physical constraints)
            X_units = [[1, 0, 0] for _ in feature_names]  # [kg, m, s] dummy
            y_units = [1, 0, 0]

            # Run SR
            # Note: run_config / config selection may vary across physo versions; using config0 is common.
            expression, logs = physo.SR(
                X_torch, y_torch,
                X_names=feature_names,
                X_units=X_units,
                y_name=target_name,
                y_units=y_units,
                op_names=["add", "mul", "sub", "div", "sin", "cos", "exp", "log", "sqrt", "neg"],
                run_config=getattr(physo.config, "config0", getattr(physo.config, "default", None)).config0
                if hasattr(getattr(physo, "config", None), "config0") else getattr(physo, "config", None),
                parallel_mode=False,
                epochs=max(1, n_iterations // 5)
            )

            # Expression might be a list-like or a single Program
            best_expr = expression[0] if hasattr(expression, '__len__') and len(expression) > 0 else expression

            # === Robust extraction for multiple PhySO versions ===
            equation = None
            str_formula = None

            # 1) Old API: get_infix(), get_infix_sympy()
            if hasattr(best_expr, "get_infix"):
                try:
                    str_formula = best_expr.get_infix()
                except Exception:
                    str_formula = str(best_expr)
                try:
                    sympy_expr = best_expr.get_infix_sympy()
                    equation = sympy_expr[0] if isinstance(sympy_expr, (list, tuple)) else sympy_expr
                except Exception:
                    # Try to sympify the string fallback
                    try:
                        equation = sp.sympify(str_formula)
                    except Exception:
                        equation = None

            # 2) Newer API: sympy() and/or infix()
            if equation is None and hasattr(best_expr, "sympy"):
                try:
                    equation = best_expr.sympy()
                except Exception:
                    equation = None
                try:
                    if str_formula is None and hasattr(best_expr, "infix"):
                        str_formula = best_expr.infix()
                except Exception:
                    if str_formula is None:
                        str_formula = str(best_expr)

            # 3) As a last resort, try stringification -> sympify
            if equation is None:
                # Convert object to string and attempt sympify
                if str_formula is None:
                    try:
                        str_formula = str(best_expr)
                    except Exception:
                        str_formula = None
                if str_formula:
                    try:
                        equation = sp.sympify(str_formula)
                    except Exception:
                        equation = None

            # If still no valid equation, gracefully fallback to linear model instead of crashing
            if equation is None:
                # Log a helpful message inside the exception for debugging
                raise RuntimeError("Could not extract a SymPy expression from PhySO Program object.")

            # Evaluate predictions
            y_pred_list = []
            for i in range(len(X_arr)):
                row_dict = {sp.Symbol(name): float(X_arr[i, j]) for j, name in enumerate(feature_names)}
                try:
                    pred_val = float(equation.subs(row_dict).evalf())
                    y_pred_list.append(pred_val)
                except Exception:
                    y_pred_list.append(np.nan)

            y_pred = np.array(y_pred_list)
            mask_valid = ~np.isnan(y_pred)
            score = r2_score(y_arr[mask_valid], y_pred[mask_valid]) if mask_valid.sum() > 0 else 0.0

            # Complexity handling
            complexity = getattr(best_expr, "complexity", max_complexity)
            try:
                complexity = int(complexity)
            except Exception:
                complexity = max_complexity

            return {
                "equation": equation,
                "str_formula": str_formula if str_formula is not None else str(equation),
                "score": float(score),
                "complexity": int(complexity),
                "feature_names": feature_names,
                "target_name": target_name,
                "is_linear": False,
                "method": "PhySO (GA-based)"
            }

        except Exception as e:
            # If PhySO fails to produce a usable symbolic expression, fallback to linear model
            try:
                fallback = _linear_fallback(X_arr, y_arr, feature_names, target_name)
                # annotate the method that PhySO failed
                fallback["method"] = f"PhySO failed -> {fallback['method']}"
                # include the original exception message in 'str_formula' or a details field
                fallback["str_formula"] = f"{fallback['str_formula']}  # Note: PhySO failed with: {e}"
                return fallback
            except Exception as e2:
                # If the linear fallback somehow also fails, raise explicit error
                raise FormulaDiscoveryError(f"PhySO failed: {e}; Linear fallback also failed: {e2}")

    # === Linear fallback requested explicitly ===
    if method == "linear" and linear_available:
        return _linear_fallback(X_arr, y_arr, feature_names, target_name)

    # If the requested method isn't available
    raise FormulaDiscoveryError(f"Method '{method}' not available.")


# === Data loading / preprocessing ===
def load_and_preprocess_data(uploaded_files, n_rows=None):
    """Load numeric Excel data or generate a sample if no files provided."""
    if not uploaded_files:
        st.info("Using sample data.")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'Feature1': rng.normal(1.2, 0.05, 200),
            'Feature2': rng.normal(500, 50, 200),
            'Feature3': rng.normal(30, 2, 200),
            'Target': 2 * rng.normal(1.2, 0.05, 200) + np.sin(rng.normal(30, 2, 200))
        })
        return df

    dfs = []
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            df_temp = pd.read_excel(io.BytesIO(uploaded_file.read()), engine='openpyxl')
            numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                df_temp = df_temp[numeric_cols].fillna(df_temp[numeric_cols].median())
                if n_rows:
                    df_temp = df_temp.sample(n=min(n_rows, len(df_temp)), random_state=42)
                dfs.append(df_temp)
        except Exception:
            # Skip unreadable files but continue processing others
            continue

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
    else:
        st.error("No numeric data found in uploaded files.")

if 'df' not in st.session_state:
    st.warning("⚠️ Load data first (or click 'Load Data' to use sample data).")
    st.stop()

df = st.session_state.df.copy()
params = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    formula_features = st.multiselect("🔧 Select Features", options=params, default=params[:-1] if len(params) > 1 else [])
with col2:
    formula_target = st.selectbox("🎯 Target Variable", options=params, index=len(params)-1 if params else 0)

if not formula_features or formula_target not in params or formula_target in formula_features:
    st.error("❌ Select valid features (excluding target).")
    st.stop()

# Available methods
available_methods = []
if pysr_available:
    available_methods.append("pysr")
if physo_available:
    available_methods.append("physo")
available_methods.append("linear")

method_options = {
    "pysr": "PySR (Evolutionary Symbolic Regression)",
    "physo": "PhySO (GA-based Symbolic Regression)",
    "linear": "Linear Regression"
}

selected_method_key = st.radio(
    "📊 Select Method",
    options=available_methods,
    format_func=lambda key: method_options.get(key, key),
    index=0,
    help="Choose the discovery method: Evolutionary/GA for nonlinear (complex formulas with exp, trig, etc.), Linear for simple fits."
)

run_formula = st.button("🚀 Discover Formula", type="primary")

if run_formula:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📊 Preparing data...")
        progress_bar.progress(0.1)

        X_formula = df[formula_features].copy()
        y_formula = df[formula_target].copy()

        # Remove rows with NaNs
        mask = ~(X_formula.isna().any(axis=1) | y_formula.isna())
        X_formula = X_formula[mask]
        y_formula = y_formula[mask]

        if len(X_formula) < min_rows:
            raise FormulaDiscoveryError(f"Insufficient valid data: {len(X_formula)} rows (need ≥{min_rows})")

        progress_bar.progress(0.2)
        status_text.text(f"🔍 Running {method_options[selected_method_key]}...")

        formula_result = discover_formula(
            X_formula, y_formula, formula_features,
            max_complexity=max_complexity,
            n_iterations=n_iterations,
            target_name=formula_target,
            method=selected_method_key
        )

        progress_bar.progress(0.7)
        status_text.text("📈 Generating visualization...")

        fallback_msg = " (Linear Approximation)" if formula_result.get("is_linear") else ""
        st.success(f"✅ Formula discovered with {formula_result['method']}{fallback_msg}!")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("📊 R² Score", f"{formula_result['score']:.4f}")
        with col_res2:
            st.metric("🔢 Complexity", formula_result['complexity'])

        st.subheader("📜 Discovered Formula")
        st.info(f"**Method:** {formula_result['method']}")

        # Try to render as LaTeX; protect against non-sympy types
        try:
            latex_formula = sp.latex(formula_result['equation'])
            st.latex(latex_formula)
        except Exception:
            # Fallback: show plain text
            st.code(formula_result.get('str_formula', str(formula_result.get('equation'))), language='text')

        with st.expander("🔤 Plain Text Version"):
            st.code(formula_result.get('str_formula', str(formula_result.get('equation'))), language='text')

        with st.expander("ℹ️ Details"):
            st.write(f"**Target:** {formula_result['target_name']}")
            st.write(f"**Features Used:** {', '.join(formula_result['feature_names'])}")

        # Evaluate equation on the dataset for plotting
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
                title=f'Predictions vs. Actual Values ({formula_result["method"]})',
            )
            min_val = float(min(y_actual_valid.min(), y_pred_valid.min()))
            max_val = float(max(y_actual_valid.max(), y_pred_valid.max()))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Fit',
                line=dict(dash='dash', color='red', width=2)
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Could not evaluate formula on data points — predictions contain NaNs.")

        # Downloadable report
        formula_text = f"""Formula Discovery Results
===================
Target: {formula_result['target_name']}
Method: {formula_result['method']}
R² Score: {formula_result['score']:.4f}
Complexity: {formula_result['complexity']}
Features: {', '.join(formula_result['feature_names'])}

Plain Text:
{formula_result.get('str_formula', str(formula_result.get('equation')))}
"""
        st.download_button(
            "💾 Download Report",
            formula_text,
            f"formula_report_{formula_result['target_name']}.txt",
            "text/plain"
        )

        progress_bar.progress(1.0)
        status_text.text("✅ Done!")
    except FormulaDiscoveryError as fe:
        st.error(f"❌ Discovery Error: {fe}")
    except Exception as e:
        st.error(f"💥 Unexpected Error: {e}")
    finally:
        try:
            progress_bar.empty()
        except Exception:
            pass
        try:
            status_text.empty()
        except Exception:
            pass
