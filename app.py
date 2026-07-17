"""
Standalone Streamlit App for Formula Discovery.
Supports PySR, AI Feynman, SINDy (dynamical systems), PolynomialFeatures,
Nonlinear Curve Fitting (scipy), and Linear Regression.
Run with: streamlit run formula_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import tempfile
from typing import List, Dict, Any
import sympy as sp
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.optimize import curve_fit
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

sindy_available = False
try:
    import pysindy as ps
    sindy_available = True
except Exception:
    pass

feynman_available = False
try:
    from aifeynman import run_aifeynman
    feynman_available = True
except Exception:
    pass

linear_available = True        # Always available
poly_available = True           # PolynomialFeatures is always available with sklearn
curve_fit_available = True      # scipy.optimize.curve_fit is always available


class FormulaDiscoveryError(Exception):
    pass


@st.cache_data
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
    custom_model: str = None,
    feynman_time_budget: int = 60
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

    # === AI Feynman (physics-inspired symbolic regression) ===
    # NOTE: this is a best-effort integration. AI Feynman is a heavy external
    # package (brute force + neural-net-guided search) that reads/writes files
    # on disk and can take minutes. Its output format can vary slightly by
    # package version, so the parsing below may need small tweaks for your
    # installed version.
    if method == "feynman" and feynman_available:
        try:
            tmpdir = tempfile.mkdtemp()
            data_fname = "data.txt"
            data_arr = np.column_stack([X_arr, y_arr])
            np.savetxt(os.path.join(tmpdir, data_fname), data_arr)

            run_aifeynman(
                tmpdir + os.sep,
                data_fname,
                BF_try_time=feynman_time_budget,
                BF_ops_file_type="14ops",
                polyfit_deg=min(poly_degree, 4),
                NN_epochs=100
            )

            results_dir = os.path.join(tmpdir, "results")
            solution_path = os.path.join(results_dir, f"solution_{data_fname}")
            if not os.path.exists(solution_path):
                raise FormulaDiscoveryError(
                    "AI Feynman did not produce a solution file. "
                    "Try increasing the time budget or check your aifeynman version's output layout."
                )

            with open(solution_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            # Each line is typically: complexity, error/MDL, expression (format varies by version)
            def parse_error(line):
                parts = line.split()
                try:
                    return float(parts[1])
                except Exception:
                    return float("inf")

            best_line = sorted(lines, key=parse_error)[0]
            eq_str = best_line.split(",")[-1].strip()

            var_map = {f"x{i}": name for i, name in enumerate(feature_names)}
            for old, new in var_map.items():
                eq_str = eq_str.replace(old, new)

            equation = sp.sympify(eq_str)
            complexity = len(list(sp.preorder_traversal(equation)))

            y_pred = np.array([
                float(equation.subs({sp.Symbol(n): X_arr[i, j] for j, n in enumerate(feature_names)}).evalf())
                for i in range(len(X_arr))
            ])
            score = r2_score(y_arr, y_pred)

            return {
                "equation": equation,
                "str_formula": str(sp.simplify(equation)),
                "score": float(score),
                "complexity": int(complexity),
                "feature_names": feature_names,
                "target_name": target_name,
                "is_linear": False,
                "method": "AI Feynman"
            }
        except FormulaDiscoveryError:
            raise
        except Exception as e:
            raise FormulaDiscoveryError(f"AI Feynman failed: {e}")

    # === Polynomial Regression (PolynomialFeatures + LinearRegression) ===
    if method == "poly" and poly_available:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_arr)

            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)

            model = LinearRegression()
            model.fit(X_poly, y_arr)
            y_pred = model.predict(X_poly)
            score = r2_score(y_arr, y_pred)

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
                "method": f"Polynomial Regression (Degree {poly_degree})",
                "scaler": scaler,
                "poly": poly,
                "model": model
            }
        except Exception as e:
            raise FormulaDiscoveryError(f"Polynomial Regression failed: {e}")

    # === Nonlinear Curve Fitting (scipy.optimize.curve_fit) ===
    if method == "curve_fit" and curve_fit_available:
        try:
            model_templates = {
                "exponential": lambda X, a, b, c: a * np.exp(b * X[:, 0]) + c,
                "sinusoidal": lambda X, a, b, c: a * np.sin(b * X[:, 0]) + c,
                "logistic": lambda X, a, b, c: a / (1 + np.exp(-b * (X[:, 0] - c))),
                "power_law": lambda X, a, b, c: a * X[:, 0]**b + c
            }

            params = None
            if nonlinear_model == "custom" and custom_model:
                try:
                    expr = sp.sympify(custom_model)
                    feature_symbols = [sp.Symbol(name) for name in feature_names]
                    # FIX: previously this substituted whole numpy arrays into a
                    # sympy expression one call at a time and cast the result to
                    # a single float, which cannot work for more than one data
                    # row (curve_fit always calls the model with the full array).
                    # lambdify gives a properly vectorized numpy function instead.
                    param_symbols = sorted(
                        [s for s in expr.free_symbols if str(s) not in feature_names + [target_name]],
                        key=str
                    )
                    func_lambdified = sp.lambdify(feature_symbols + param_symbols, expr, modules="numpy")

                    def custom_func(X, *p, _f=func_lambdified, _nfeat=len(feature_names)):
                        col_args = [X[:, i] for i in range(_nfeat)]
                        result = _f(*col_args, *p)
                        # broadcast in case the expression doesn't depend on every column
                        return np.broadcast_to(np.asarray(result, dtype=np.float64), (X.shape[0],))

                    model_func = custom_func
                    params = [str(s) for s in param_symbols]
                    n_params = len(params)
                except Exception as e:
                    raise FormulaDiscoveryError(f"Invalid custom model: {e}")
            else:
                model_func = model_templates.get(nonlinear_model)
                n_params = 3
                if not model_func:
                    raise FormulaDiscoveryError(f"Unknown model: {nonlinear_model}")

            popt, _ = curve_fit(model_func, X_arr, y_arr, p0=np.ones(n_params), maxfev=n_iterations * 100)
            y_pred = model_func(X_arr, *popt)
            score = r2_score(y_arr, y_pred)

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
                elif nonlinear_model == "power_law":
                    equation = popt[0] * x0**popt[1] + popt[2]
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
            "method": "Linear Regression",
            "model": model
        }
    raise FormulaDiscoveryError(f"Method '{method}' not available.")


@st.cache_data
def discover_dynamics_sindy(
    df_states: pd.DataFrame,
    state_names: List[str],
    dt: float = 1.0,
    time_values: List[float] = None,
    poly_degree: int = 2,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Fit a dynamical system dx/dt = f(x) from time-series data using SINDy
    (Sparse Identification of Nonlinear Dynamics). Unlike the other methods
    above, this does not predict a single target from features - it discovers
    a coupled ODE system describing how ALL selected state variables evolve.
    """
    X = df_states[state_names].values.astype(np.float64)

    if len(X) < 5:
        raise FormulaDiscoveryError("Need at least 5 time points for SINDy.")
    if np.any(np.isnan(X)):
        raise FormulaDiscoveryError("Invalid data: NaNs found in selected state columns.")

    t_arg = np.asarray(time_values, dtype=np.float64) if time_values is not None else dt

    try:
        model = ps.SINDy(
            feature_names=state_names,
            feature_library=ps.PolynomialLibrary(degree=poly_degree),
            optimizer=ps.STLSQ(threshold=threshold)
        )
        model.fit(X, t=t_arg)
        equations = model.equations()
        score = model.score(X, t=t_arg)
    except Exception as e:
        raise FormulaDiscoveryError(f"SINDy failed: {e}")

    return {
        "equations": equations,   # list of strings, one per state variable: "x0' = ..."
        "state_names": state_names,
        "score": float(score),
        "model": model,
        "method": "SINDy (Sparse Identification of Nonlinear Dynamics)"
    }


@st.cache_data
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
        # FIX: engine='openpyxl' was hardcoded, but openpyxl cannot read legacy
        # .xls files (only .xlsx/.xlsm) - any .xls upload used to crash here.
        # Pick the engine based on the file extension instead.
        fname_lower = uploaded_file.name.lower()
        engine = "xlrd" if fname_lower.endswith(".xls") else "openpyxl"
        try:
            df_temp = pd.read_excel(io.BytesIO(uploaded_file.read()), engine=engine)
        except Exception as e:
            st.warning(
                f"⚠️ Could not read '{uploaded_file.name}': {e}. "
                f"If this is a legacy .xls file, make sure the 'xlrd' package is installed "
                f"(`pip install xlrd`), or re-save it as .xlsx."
            )
            continue

        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df_temp = df_temp[numeric_cols].fillna(df_temp[numeric_cols].median())
            if n_rows:
                df_temp = df_temp.sample(n=min(n_rows, len(df_temp)), random_state=42)
            dfs.append(df_temp)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def evaluate_formula(equation, X, feature_names, scaler=None, poly=None, model=None, method=""):
    """Evaluate formula on given data."""
    if method == "poly" and scaler is not None and poly is not None and model is not None:
        X_scaled = scaler.transform(X.values)
        X_poly = poly.transform(X_scaled)
        return model.predict(X_poly)
    else:
        y_pred = []
        for idx in range(len(X)):
            row = X.iloc[idx]
            val_dict = {sp.Symbol(name): float(row[name]) for name in feature_names}
            try:
                pred_val = float(equation.subs(val_dict).evalf())
                y_pred.append(pred_val)
            except Exception:
                y_pred.append(np.nan)
        return np.array(y_pred)


# === Streamlit UI ===
st.set_page_config(page_title="Formula Discovery App", layout="wide")

st.title("🧮 Formula Discovery App")
st.markdown("""
Welcome to the Formula Discovery App! Upload an Excel file with numeric data, select features and a target variable,
and choose a method to discover a mathematical formula. After discovering a formula, you can validate it on new Excel data
or edit the formula interactively. Use the sidebar to configure settings.

**Excel format expected:** first sheet only, numeric columns with a header row; non-numeric columns are dropped
automatically; missing values are filled with the column median.
""")

st.sidebar.header("⚙️ Config")
st.sidebar.markdown("Adjust parameters for formula discovery.")
n_iterations = st.sidebar.number_input("Iterations", min_value=10, value=100, help="Number of search iterations for PySR or curve fitting")
max_complexity = st.sidebar.number_input("Max Complexity", min_value=1, value=10, help="Maximum equation size for PySR")
min_rows = st.sidebar.number_input("Min Rows", min_value=5, value=10, help="Minimum data points required")
poly_degree = st.sidebar.number_input("Polynomial Degree", min_value=1, value=2, help="Degree for Polynomial Regression / SINDy library / AI Feynman poly-fit fallback")
nonlinear_model = st.sidebar.selectbox(
    "Nonlinear Model",
    options=["exponential", "sinusoidal", "logistic", "power_law", "custom"],
    help="Model type for Nonlinear Curve Fitting (e.g., exponential: a*e^(b*x)+c, power_law: a*x^b+c)"
)
custom_model = st.sidebar.text_input(
    "Custom Model (sympy syntax)",
    value="",
    help="Enter a custom model, e.g., 'a * x1 + b * sin(x2) + c'. Leave blank for predefined models."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**SINDy settings** (used only if SINDy is selected)")
sindy_threshold = st.sidebar.number_input("SINDy Sparsity Threshold", min_value=0.0, value=0.1, step=0.01, help="Higher = sparser (simpler) discovered equations")
sindy_dt = st.sidebar.number_input("SINDy Time Step (dt)", min_value=0.0001, value=1.0, help="Used if no explicit time column is selected")

st.sidebar.markdown("**AI Feynman settings** (used only if AI Feynman is selected)")
feynman_time_budget = st.sidebar.number_input("Feynman Time Budget (sec)", min_value=10, value=60, help="Brute-force search time budget per pass")

uploaded_files = st.file_uploader("📁 Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'], help="Upload one or more Excel files with numeric data.")
n_rows_input = st.number_input("Sample rows (0 for all)", min_value=0, value=0, help="Number of rows to sample (0 to use all)")
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

# Available methods
available_methods = []
if pysr_available:
    available_methods.append("pysr")
if feynman_available:
    available_methods.append("feynman")
available_methods.extend(["poly", "curve_fit", "linear"])
if sindy_available:
    available_methods.append("sindy")

method_options = {
    "pysr": "PySR (Evolutionary Symbolic Regression): Finds complex formulas using genetic algorithms",
    "feynman": "AI Feynman: Physics-inspired brute-force + neural-net-guided symbolic regression",
    "poly": f"Polynomial Regression (Degree {poly_degree}): Fits a polynomial of specified degree",
    "curve_fit": f"Nonlinear Curve Fitting ({nonlinear_model}): Fits a specific nonlinear model",
    "linear": "Linear Regression: Fits a simple linear model",
    "sindy": "SINDy: Discovers a dynamical system (dx/dt = f(x)) from time-series state variables"
}

if not pysr_available:
    st.sidebar.caption("ℹ️ PySR not installed — `pip install pysr` to enable it.")
if not feynman_available:
    st.sidebar.caption("ℹ️ AI Feynman not installed — `pip install aifeynman` to enable it (heavy dependency, needs a Fortran compiler).")
if not sindy_available:
    st.sidebar.caption("ℹ️ SINDy not installed — `pip install pysindy` to enable it.")

selected_method_key = st.radio(
    "📊 Select Method",
    options=available_methods,
    format_func=lambda key: method_options[key],
    index=0,
    help="Choose a method based on your data and goal."
)

# =========================================================================
# SINDy has a fundamentally different input shape (state variables over
# time, not features -> single target), so it gets its own panel.
# =========================================================================
if selected_method_key == "sindy":
    st.subheader("🌀 Dynamical Systems Discovery (SINDy)")
    st.markdown(
        "Select the columns that represent the **state variables of a dynamical system**, "
        "measured at successive time steps (rows must already be in time order)."
    )

    state_vars = st.multiselect("🔧 Select State Variables", options=params, default=params[:min(2, len(params))])
    use_time_col = st.checkbox("Use an explicit time column instead of a fixed dt", value=False)

    time_col = None
    if use_time_col:
        time_col = st.selectbox("🕒 Time Column", options=[c for c in params if c not in state_vars])

    run_sindy = st.button("🚀 Discover Dynamical System", type="primary")

    if run_sindy:
        if len(state_vars) < 1:
            st.error("❌ Select at least one state variable.")
            st.stop()
        try:
            df_states = df.dropna(subset=state_vars).reset_index(drop=True)
            if len(df_states) < min_rows:
                raise FormulaDiscoveryError(f"Insufficient valid data: {len(df_states)} rows (need ≥{min_rows})")

            time_values = df_states[time_col].values.tolist() if time_col else None

            sindy_result = discover_dynamics_sindy(
                df_states, state_vars,
                dt=sindy_dt,
                time_values=time_values,
                poly_degree=poly_degree,
                threshold=sindy_threshold
            )

            st.success(f"✅ Dynamical system discovered with {sindy_result['method']}!")
            st.metric("📊 Overall R² Score", f"{sindy_result['score']:.4f}")

            st.subheader("📜 Discovered Equations")
            for eq in sindy_result["equations"]:
                st.code(eq, language="text")

            # Simulate/compare against the actual trajectory for a quick visual check
            try:
                X_states = df_states[state_vars].values.astype(np.float64)
                t_arg = np.asarray(time_values, dtype=np.float64) if time_values else sindy_dt
                x_dot_actual = sindy_result["model"].differentiate(X_states, t=t_arg)
                x_dot_pred = sindy_result["model"].predict(X_states)

                for i, name in enumerate(state_vars):
                    fig = px.scatter(
                        x=x_dot_actual[:, i], y=x_dot_pred[:, i],
                        labels={'x': f'Actual d{name}/dt', 'y': f'Predicted d{name}/dt'},
                        title=f'SINDy fit check: d{name}/dt'
                    )
                    lo = min(x_dot_actual[:, i].min(), x_dot_pred[:, i].min())
                    hi = max(x_dot_actual[:, i].max(), x_dot_pred[:, i].max())
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect Fit',
                                              line=dict(dash='dash', color='red', width=2)))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not render derivative fit plots: {e}")

        except FormulaDiscoveryError as e:
            st.error(f"❌ Discovery Error: {str(e)}")
        except Exception as e:
            st.error(f"💥 Unexpected Error: {str(e)}")

    st.stop()

# =========================================================================
# All other methods share the original features -> single target flow
# =========================================================================
col1, col2 = st.columns(2)
with col1:
    formula_features = st.multiselect("🔧 Select Features", options=params, default=params[:-1] if len(params) > 1 else [], help="Select input variables for the formula")
with col2:
    formula_target = st.selectbox("🎯 Target Variable", options=params, help="Select the variable to predict")

if not formula_features or formula_target not in params or formula_target in formula_features:
    st.error("❌ Select valid features (excluding target).")
    st.stop()

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
            custom_model=custom_model,
            feynman_time_budget=feynman_time_budget
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

        # Compute predictions for plotting
        y_pred = evaluate_formula(
            formula_result['equation'], X_formula, formula_result['feature_names'],
            scaler=formula_result.get("scaler"), poly=formula_result.get("poly"), model=formula_result.get("model"),
            method=selected_method_key
        )

        mask_valid = ~np.isnan(y_pred)
        if mask_valid.sum() > 0:
            y_actual_valid = y_formula.values[mask_valid]
            y_pred_valid = y_pred[mask_valid]

            # Scatter plot
            fig = px.scatter(
                x=y_actual_valid, y=y_pred_valid,
                labels={'x': f'Actual {formula_target}', 'y': f'Predicted {formula_target}'},
                title=f'Predictions vs. Actual Values ({formula_result["method"]})'
            )
            min_val = min(y_actual_valid.min(), y_pred_valid.min())
            max_val = max(y_actual_valid.max(), y_pred_valid.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Fit',
                line=dict(dash='dash', color='red', width=2)
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot
            residuals = y_actual_valid - y_pred_valid
            fig_res = px.scatter(
                x=y_actual_valid, y=residuals,
                labels={'x': f'Actual {formula_target}', 'y': 'Residuals'},
                title=f'Residuals vs. Actual Values ({formula_result["method"]})'
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
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

        # Interactive formula editor
        st.subheader("✏️ Edit Formula")
        edited_formula = st.text_input(
            "Edit Formula (sympy syntax)",
            value=formula_result['str_formula'],
            help="Modify the formula (e.g., change coefficients) and see updated predictions."
        )
        if st.button("🔄 Evaluate Edited Formula"):
            try:
                edited_eq = sp.sympify(edited_formula)
                y_pred_edited = evaluate_formula(
                    edited_eq, X_formula, formula_result['feature_names'],
                    method=""  # always symbolic substitution for a freshly-edited formula
                )
                mask_valid_edited = ~np.isnan(y_pred_edited)
                if mask_valid_edited.sum() > 0:
                    y_actual_valid_edited = y_formula.values[mask_valid_edited]
                    y_pred_valid_edited = y_pred_edited[mask_valid_edited]
                    edited_score = r2_score(y_actual_valid_edited, y_pred_valid_edited)
                    st.metric("📊 Edited Formula R² Score", f"{edited_score:.4f}")

                    fig_edited = px.scatter(
                        x=y_actual_valid_edited, y=y_pred_valid_edited,
                        labels={'x': f'Actual {formula_target}', 'y': f'Predicted {formula_target}'},
                        title='Edited Formula: Predictions vs. Actual Values'
                    )
                    min_val = min(y_actual_valid_edited.min(), y_pred_valid_edited.min())
                    max_val = max(y_actual_valid_edited.max(), y_pred_valid_edited.max())
                    fig_edited.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='Perfect Fit',
                        line=dict(dash='dash', color='red', width=2)
                    ))
                    st.plotly_chart(fig_edited, use_container_width=True)
                else:
                    st.warning("⚠️ Could not evaluate edited formula.")
            except Exception as e:
                st.error(f"❌ Invalid formula syntax: {str(e)}")

        # Store formula result for validation
        st.session_state.formula_result = formula_result
        st.session_state.selected_method_key = selected_method_key

        progress_bar.progress(1.0)
    except FormulaDiscoveryError as e:
        st.error(f"❌ Discovery Error: {str(e)}")
    except Exception as e:
        st.error(f"💥 Unexpected Error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

# Validation module
if 'formula_result' in st.session_state:
    validation_placeholder = st.empty()
    if validation_placeholder.button("🛡️ Validate Formula on New Data"):
        st.session_state.show_validation_uploader = True

    if st.session_state.get('show_validation_uploader', False):
        with validation_placeholder.container():
            st.markdown("### Validate Formula")
            validation_files = st.file_uploader(
                "📁 Upload Excel file for Validation",
                accept_multiple_files=True,
                type=['xlsx', 'xls'],
                key='validation_uploader',
                help="Upload an Excel file with the same features and target as the original data."
            )
            if validation_files:
                try:
                    val_df = load_and_preprocess_data(validation_files)
                    if val_df.empty:
                        raise ValueError("No valid numeric data found in uploaded Excel file.")

                    formula_result = st.session_state.formula_result
                    selected_method_key = st.session_state.selected_method_key
                    required_cols = formula_result['feature_names'] + [formula_result['target_name']]

                    # Column mapping for validation data
                    st.write("Map columns if names differ from original data:")
                    col_mapping = {}
                    for orig_col in required_cols:
                        mapped_col = st.selectbox(
                            f"Select column for '{orig_col}'",
                            options=val_df.columns,
                            key=f"map_{orig_col}",
                            help=f"Choose the column in the validation file that corresponds to '{orig_col}'"
                        )
                        col_mapping[orig_col] = mapped_col

                    # Rename columns in validation data
                    val_df_mapped = val_df.rename(columns={v: k for k, v in col_mapping.items()})

                    if set(required_cols).issubset(val_df_mapped.columns):
                        X_val = val_df_mapped[formula_result['feature_names']]
                        y_val = val_df_mapped[formula_result['target_name']]

                        y_pred_val = evaluate_formula(
                            formula_result['equation'], X_val, formula_result['feature_names'],
                            scaler=formula_result.get("scaler"), poly=formula_result.get("poly"), model=formula_result.get("model"),
                            method=selected_method_key
                        )

                        mask_valid = ~np.isnan(y_pred_val)
                        if mask_valid.sum() > 0:
                            y_val_valid = y_val.values[mask_valid]
                            y_pred_valid = y_pred_val[mask_valid]
                            val_score = r2_score(y_val_valid, y_pred_valid)
                            st.metric("🛡️ Validation R² Score", f"{val_score:.4f}")

                            fig_val = px.scatter(
                                x=y_val_valid, y=y_pred_valid,
                                labels={'x': f'Actual {formula_result["target_name"]}', 'y': f'Predicted {formula_result["target_name"]}'},
                                title=f'Validation: Predictions vs. Actual Values ({formula_result["method"]})'
                            )
                            min_val = min(y_val_valid.min(), y_pred_valid.min())
                            max_val = max(y_val_valid.max(), y_pred_valid.max())
                            fig_val.add_trace(go.Scatter(
                                x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Fit',
                                line=dict(dash='dash', color='red', width=2)
                            ))
                            st.plotly_chart(fig_val, use_container_width=True)

                            # Validation residual plot
                            residuals_val = y_val_valid - y_pred_valid
                            fig_res_val = px.scatter(
                                x=y_val_valid, y=residuals_val,
                                labels={'x': f'Actual {formula_result["target_name"]}', 'y': 'Residuals'},
                                title=f'Validation: Residuals vs. Actual Values ({formula_result["method"]})'
                            )
                            fig_res_val.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_res_val, use_container_width=True)
                        else:
                            st.warning("⚠️ Could not evaluate formula on validation data points.")
                    else:
                        st.error(f"❌ Validation Excel file must contain columns: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"❌ Validation Error: {str(e)}")
