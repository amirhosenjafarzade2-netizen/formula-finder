"""
Standalone Streamlit App for Formula Discovery.
Supports PySR (if Julia available), PolynomialFeatures, Nonlinear Curve Fitting (scipy), Symbolic Curve Fitting (symfit), and Linear Regression.
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.optimize import curve_fit
import symfit
import uuid

# Fix Julia env (for Streamlit Cloud)
os.environ['JULIA_DEPOT_PATH'] = '/tmp/julia'
os.environ['JULIA_LOAD_PATH'] = '/tmp/julia'

# === Backend detection ===
pysr_available = False
try:
    from pysr import PySRRegressor
    pysr_available = True
except Exception:
    pass

linear_available = True  # Always available
poly_available = True    # PolynomialFeatures is always available with sklearn
curve_fit_available = True  # scipy.optimize.curve_fit is always available
symfit_available = True  # Assume symfit is installed; handle errors if not

class FormulaDiscoveryError(Exception):
    pass

def discover_formula(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    max_complexity: int = 10,
    n_iterations: int = 100,
    target_name: str = "y",
    method: str = "pysr",
    poly_degree: int = 2,
    nonlinear_model: str = "exponential",
    custom_model: str = None
) -> Dict[str, Any]:
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
                "method": "PySR (Evolutionary)"
            }
        except Exception as e:
            raise FormulaDiscoveryError(f"PySR failed: {e}")

    # === Polynomial Regression (PolynomialFeatures + LinearRegression) ===
    if method == "poly" and poly_available:
        try:
            # Scale features to stabilize polynomial regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_arr)

            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)

            model = LinearRegression()
            model.fit(X_poly, y_arr)
            y_pred = model.predict(X_poly)
            score = r2_score(y_arr, y_pred)

            # Construct symbolic equation
            feature_names_poly = poly.get_feature_names_out(feature_names)
            terms = [sp.Float(coef) * sp.sympify(name.replace(" ", "*")) for coef, name in zip(model.coef_, feature_names_poly)]
            equation = sum(terms) + sp.Float(model.intercept_)
            complexity = len(feature_names_poly) + 1

            return {
                "equation": equation,
                "str_formula": str(sp.simplify(equation)),
                "score": float(score),
                "complexity": int(complexity),
                "feature_names": feature_names,
                "target_name": target_name,
                "is_linear": False,
                "method": f"Polynomial Regression (Degree {poly_degree})"
            }
        except Exception as e:
            raise FormulaDiscoveryError(f"Polynomial Regression failed: {e}")

    # === Nonlinear Curve Fitting (scipy.optimize.curve_fit) ===
    if method == "curve_fit" and curve_fit_available:
        try:
            # Define predefined model templates
            model_templates = {
                "exponential": lambda X, a, b, c: a * np.exp(b * X[:, 0]) + c,
                "sinusoidal": lambda X, a, b, c: a * np.sin(b * X[:, 0]) + c,
                "logistic": lambda X, a, b, c: a / (1 + np.exp(-b * (X[:, 0] - c)))
            }

            # Handle custom model
            if nonlinear_model == "custom" and custom_model:
                try:
                    expr = sp.sympify(custom_model)
                    params = sorted([str(p) for p in expr.free_symbols if str(p) not in feature_names + [target_name]])
                    def custom_func(X, *p):
                        subs_dict = {sp.Symbol(feature_names[i]): X[:, i] for i in range(X.shape[1])}
                        subs_dict.update({sp.Symbol(param): p[i] for i, param in enumerate(params)})
                        return float(expr.subs(subs_dict).evalf())
                    model_func = custom_func
                    n_params = len(params)
                except Exception as e:
                    raise FormulaDiscoveryError(f"Invalid custom model: {e}")
            else:
                model_func = model_templates.get(nonlinear_model)
                n_params = 3  # a, b, c for predefined models
                if not model_func:
                    raise FormulaDiscoveryError(f"Unknown model: {nonlinear_model}")

            # Fit model
            popt, _ = curve_fit(model_func, X_arr, y_arr, maxfev=n_iterations * 100)
            y_pred = model_func(X_arr, *popt)
            score = r2_score(y_arr, y_pred)

            # Construct symbolic equation
            if nonlinear_model == "custom" and custom_model:
                equation = sp.sympify(custom_model)
                subs_dict = {sp.Symbol(param): popt[i] for i, param in enumerate(params)}
                equation = equation.subs(subs_dict)
            else:
                x0 = sp.Symbol(feature_names[0])
                if nonlinear_model == "exponential":
                    equation = popt[0] * sp.exp(popt[1] * x0) + popt[2]
                elif nonlinear_model == "sinusoidal":
                    equation = popt[0] * sp.sin(popt[1] * x0) + popt[2]
                elif nonlinear_model == "logistic":
                    equation = popt[0] / (1 + sp.exp(-popt[1] * (x0 - popt[2])))
                else:
                    raise FormulaDiscoveryError("Model not implemented.")

            complexity = len(list(sp.preorder_traversal(equation)))
            str_formula = str(sp.simplify(equation))

            return {
                "equation": equation,
                "str_formula": str_formula,
                "score": float(score),
                "complexity": int(complexity),
                "feature_names": feature_names,
                "target_name": target_name,
                "is_linear": False,
                "method": f"Nonlinear Curve Fitting ({nonlinear_model})"
            }
        except Exception as e:
            raise FormulaDiscoveryError(f"Nonlinear Curve Fitting failed: {e}")

    # === Symbolic Curve Fitting (symfit) ===
    if method == "symfit" and symfit_available:
        try:
            # Define variables and parameters
            variables = {name: symfit.Variable(name) for name in feature_names}
            params = {f'p{i}': symfit.Parameter(f'p{i}') for i in range(3)}  # Example: 3 parameters
            y_var = symfit.Variable(target_name)

            # Define model (predefined or custom)
            if nonlinear_model == "custom" and custom_model:
                try:
                    expr = sp.sympify(custom_model)
                    param_names = sorted([str(p) for p in expr.free_symbols if str(p) not in feature_names + [target_name]])
                    params = {name: symfit.Parameter(name) for name in param_names}
                    model = sp.lambdify(list(params.values()) + list(variables.values()), expr, 'numpy')
                    model_dict = {y_var: model}
                except Exception as e:
                    raise FormulaDiscoveryError(f"Invalid custom model: {e}")
            else:
                x0 = variables[feature_names[0]]
                if nonlinear_model == "exponential":
                    model_dict = {y_var: params['p0'] * symfit.exp(params['p1'] * x0) + params['p2']}
                elif nonlinear_model == "sinusoidal":
                    model_dict = {y_var: params['p0'] * symfit.sin(params['p1'] * x0) + params['p2']}
                elif nonlinear_model == "logistic":
                    model_dict = {y_var: params['p0'] / (1 + symfit.exp(-params['p1'] * (x0 - params['p2'])))}
                else:
                    raise FormulaDiscoveryError(f"Unknown model: {nonlinear_model}")

            # Fit model
            fit = symfit.Fit(model_dict, **{name: X_arr[:, i] for i, name in enumerate(feature_names)}, y=y_arr)
            fit_result = fit.execute(maxiter=n_iterations * 100)

            # Construct symbolic equation
            param_values = {sp.Symbol(k): v for k, v in fit_result.params.items()}
            equation = list(model_dict.values())[0]
            equation = sp.sympify(str(equation).replace('Variable', '').replace('Parameter', ''))
            equation = equation.subs(param_values)
            complexity = len(list(sp.preorder_traversal(equation)))
            str_formula = str(sp.simplify(equation))

            # Predict for scoring
            X_dict = {name: X_arr[:, i] for i, name in enumerate(feature_names)}
            y_pred = fit.model(**X_dict, **fit_result.params).y
            score = r2_score(y_arr, y_pred)

            return {
                "equation": equation,
                "str_formula": str_formula,
                "score": float(score),
                "complexity": int(complexity),
                "feature_names": feature_names,
                "target_name": target_name,
                "is_linear": False,
                "method": f"Symbolic Curve Fitting ({nonlinear_model})"
            }
        except Exception as e:
            raise FormulaDiscoveryError(f"Symbolic Curve Fitting failed: {e}")

    # === Linear Regression ===
    if method == "linear" and linear_available:
        model = LinearRegression()
        model.fit(X_arr, y_arr)
        y_pred = model.predict(X_arr)
        score = r2_score(y_arr, y_pred)

        coeffs = model.coef_
        intercept = model.intercept_
        terms = [sp.Float(coef) * sp.Symbol(name) for name, coef in zip(feature_names, coeffs)]
        equation = sum(terms) + sp.Float(intercept)

        return {
            "equation": equation,
            "str_formula": str(sp.simplify(equation)),
            "score": float(score),
            "complexity": len(feature_names) + 1,
            "feature_names": feature_names,
            "target_name": target_name,
            "is_linear": True,
            "method": "Linear Regression"
        }

    raise FormulaDiscoveryError(f"Method '{method}' not available.")

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
poly_degree = st.sidebar.number_input("Polynomial Degree", min_value=1, value=2, help="Degree for Polynomial Regression")
nonlinear_model = st.sidebar.selectbox(
    "Nonlinear Model",
    options=["exponential", "sinusoidal", "logistic", "custom"],
    help="Model type for Nonlinear and Symbolic Curve Fitting"
)
custom_model = st.sidebar.text_input(
    "Custom Model (sympy syntax)",
    value="",
    help="Enter a custom model, e.g., 'a * x1 + b * sin(x2) + c'. Leave blank for predefined models."
)

uploaded_files = st.file_uploader("📁 Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'])
n_rows_input = st.number_input("Sample rows (0 for all)", min_value=0, value=0)
n_rows = None if n_rows_input == 0 else n_rows_input

if st.button("Load Data"):
    df = load_and_preprocess_data(uploaded_files, n_rows)
    if not df.empty:
        st.session_state.df = df
        st.success(f"✅ Loaded {len(df)} rows with {len(df.columns)} numeric columns")
        with st.expander("👁️ Data Preview"):
            st.dataframe(df.head(10), width='stretch')

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

# Available methods
available_methods = []
if pysr_available:
    available_methods.append("pysr")
available_methods.extend(["poly", "curve_fit", "symfit", "linear"])

# Method choice
method_options = {
    "pysr": "PySR (Evolutionary Symbolic Regression)",
    "poly": f"Polynomial Regression (Degree {poly_degree})",
    "curve_fit": f"Nonlinear Curve Fitting ({nonlinear_model})",
    "symfit": f"Symbolic Curve Fitting ({nonlinear_model})",
    "linear": "Linear Regression"
}
selected_method_key = st.radio(
    "📊 Select Method",
    options=available_methods,
    format_func=lambda key: method_options[key],
    index=0,
    help="Choose the discovery method: PySR for complex formulas, Polynomial for polynomial fits, Curve/Symbolic for specific nonlinear models, Linear for simple fits."
)

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
        status_text.text(f"🔍 Running {method_options[selected_method_key]}...")

        formula_result = discover_formula(
            X_formula, y_formula, formula_features,
            max_complexity=max_complexity,
            n_iterations=n_iterations,
            target_name=formula_target,
            method=selected_method_key,
            poly_degree=poly_degree,
            nonlinear_model=nonlinear_model,
            custom_model=custom_model
        )

        progress_bar.progress(0.8)
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
                title=f'Predictions vs. Actual Values ({formula_result["method"]})',
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
Method: {formula_result['method']}
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
