import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import sympy as sp
import io
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

def validate_formula(
    formula_result: Dict,
    selected_method_key: str,
    formula_features: List[str],
    formula_target: str,
    validation_files,
    load_and_preprocess_data
):
    """
    Validate a discovered formula on new data, compute R² score, and plot predictions vs. actual values.
    """
    if not validation_files:
        st.warning("⚠️ Please upload validation Excel files.")
        return

    df_val = load_and_preprocess_data(validation_files, None)
    if df_val.empty:
        st.error("❌ Failed to load validation data.")
        return

    required_cols = set(formula_features + [formula_target])
    if required_cols - set(df_val.columns):
        st.error(f"❌ Validation data missing columns: {required_cols - set(df_val.columns)}")
        return

    st.success(f"✅ Loaded validation data with {len(df_val)} rows")
    X_val = df_val[formula_features]
    y_val = df_val[formula_target]

    if selected_method_key == "poly":
        scaler = formula_result.get("scaler")
        poly = formula_result.get("poly")
        model = formula_result.get("model")
        X_scaled = scaler.transform(X_val.values)
        X_poly = poly.transform(X_scaled)
        y_pred_val = model.predict(X_poly)
    else:
        equation = formula_result['equation']
        y_pred_val = []
        for idx in range(len(X_val)):
            row = X_val.iloc[idx]
            val_dict = {sp.Symbol(name): float(row[name]) for name in formula_features}
            try:
                pred_val = float(equation.subs(val_dict).evalf())
                y_pred_val.append(pred_val)
            except Exception:
                y_pred_val.append(np.nan)
        y_pred_val = np.array(y_pred_val)

    mask_valid = ~np.isnan(y_pred_val) & ~y_val.isna()
    if mask_valid.sum() > 0:
        y_actual_val = y_val[mask_valid].values
        y_pred_val_valid = y_pred_val[mask_valid]
        score_val = r2_score(y_actual_val, y_pred_val_valid)
        st.metric("📊 Validation R² Score", f"{score_val:.4f}")

        fig_val = px.scatter(
            x=y_actual_val, y=y_pred_val_valid,
            labels={'x': f'Actual {formula_target}', 'y': f'Predicted {formula_target}'},
            title=f'Validation: Predictions vs. Actual Values ({formula_result["method"]})',
        )
        min_val = min(y_actual_val.min(), y_pred_val_valid.min())
        max_val = max(y_actual_val.max(), y_pred_val_valid.max())
        fig_val.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Fit',
            line=dict(dash='dash', color='red', width=2)
        ))
        st.plotly_chart(fig_val, use_container_width=True)
    else:
        st.warning("⚠️ Could not evaluate formula on validation data points.")
