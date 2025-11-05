import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Alloy Property Predictor", layout="wide")
st.title("🧪 Alloy Property Predictor & Alloy Type Helper")
st.caption("Models are saved to disk automatically (saved_model.pkl).")

# ---------------- Helper Functions ----------------
def suggest_alloy_type(comp):
    comp = {k.strip().lower(): float(v) for k, v in comp.items() if v not in [None, "", "-"]}
    def p(el): return comp.get(el, 0.0)
    fe, al, cu, ni, ti, cr, mn, si, c, zn, sn = map(p, ['fe','al','cu','ni','ti','cr','mn','si','c','zn','sn'])
    if fe > 40 and cr >= 10.5:
        if ni >= 8: return 'Austenitic stainless steel', 'Fe>40%, Cr≥10.5%, Ni≥8%'
        return 'Stainless steel (ferritic/martensitic)', 'Fe>40%, Cr≥10.5%'
    if fe > 40 and c > 0.05: return 'Carbon/alloy steel', f'Fe={fe}%, C={c}%'
    if fe > 40 and (mn > 0 or si > 0 or cr > 0 or ni > 0): return 'Alloy steel', 'Fe-dominant alloy'
    if al > 40: return 'Aluminium alloy', f'Al={al}%'
    if cu > 40 and zn > 0: return 'Brass (Cu-Zn)', f'Cu={cu}%, Zn={zn}%'
    if cu > 50 and sn > 0: return 'Bronze (Cu-Sn)', f'Cu={cu}%, Sn={sn}%'
    if ti > 40: return 'Titanium alloy', f'Ti={ti}%'
    if ni > 40: return 'Nickel alloy', f'Ni={ni}%'
    if comp:
        el, pct = max(comp.items(), key=lambda x:x[1])
        if pct >= 30: return f'{el.capitalize()}-rich alloy', f'{el.upper()} ~ {pct}%'
    return 'Unknown/mixed', 'No clear dominant element found'

def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    st.pyplot(fig)

def create_synthetic_dataset(path="synthetic_alloy_data.csv", n=200):
    np.random.seed(0)
    df = pd.DataFrame({
        'Cu': np.random.uniform(1.0, 3.0, n),
        'Zn': np.random.uniform(4.0, 8.0, n),
        'Mg': np.random.uniform(1.0, 3.0, n),
        'Cr': np.random.uniform(0.1, 0.3, n),
        'Ti': np.random.uniform(0.05, 0.2, n),
        'Aging_Temp_C': np.random.uniform(120, 180, n),
        'Aging_Time_h': np.random.uniform(4, 24, n),
        'Quench_Rate_Cps': np.random.uniform(50, 300, n)
    })
    df['Yield_Strength_MPa'] = 200 + 30*df['Cu'] + 20*df['Zn'] + 10*df['Mg'] + np.random.normal(0, 10, n)
    df['Hardness_HV'] = 60 + 5*df['Cu'] + 3*df['Zn'] + 2*df['Mg'] + np.random.normal(0, 5, n)
    df.to_csv(path, index=False)
    return df

# ---------------- File Upload or Synthetic Data ----------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv","txt"])

# ✅ FIX: Store dataframe persistently in Streamlit session_state
if "df" not in st.session_state:
    st.session_state["df"] = None

if st.button("Use synthetic dataset"):
    st.session_state["df"] = create_synthetic_dataset()
    st.success("✅ Synthetic dataset created and loaded.")

if uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.success("✅ CSV loaded successfully.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

df = st.session_state["df"]

# ---------------- Tabs ----------------
tabs = st.tabs(["Data", "EDA", "Model Training", "Predict", "Alloy Type Helper"])

# ---------------- Data Tab ----------------
with tabs[0]:
    st.header("📊 Data Preview")
    if df is None:
        st.info("Upload or generate dataset first.")
    else:
        st.dataframe(df.head(200))
        st.write("Summary statistics:")
        st.write(df.describe(include='all'))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        st.subheader("Select Features and Targets")
        features_selected = st.multiselect("Feature columns:", numeric_cols, default=numeric_cols[:-2])
        reg_target = st.multiselect("Numeric target(s) for regression:", numeric_cols, default=numeric_cols[-2:])
        class_target = st.selectbox("Categorical target (for classification):", [None]+cat_cols)

# ---------------- EDA Tab ----------------
with tabs[1]:
    st.header("📈 EDA")
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Need at least two numeric columns for correlation heatmap.")

# ---------------- Model Training Tab ----------------
with tabs[2]:
    st.header("🧠 Model Training (with Save/Load)")
    if df is None:
        st.info("Upload or create dataset first.")
    elif not features_selected:
        st.warning("Select features in Data tab first.")
    else:
        task_choice = st.selectbox("Task type:", ["Regression", "Classification"], index=0)
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
        model_path = "saved_model.pkl"

        if st.button("🚀 Train Model"):
            if task_choice == "Regression" and reg_target:
                data = df[features_selected + reg_target].dropna()
                X, y = data[features_selected], data[reg_target]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                if len(reg_target) > 1:
                    model = MultiOutputRegressor(model)
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                joblib.dump(pipe, model_path)
                st.success("✅ Regression model trained and saved!")

                if preds.ndim == 1:
                    st.json({
                        'R2': r2_score(y_test, preds),
                        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
                        'MAE': mean_absolute_error(y_test, preds)
                    })
                    plot_actual_vs_pred(y_test, preds)
                else:
                    for i, col in enumerate(reg_target):
                        st.subheader(f"Target: {col}")
                        st.json({
                            'R2': r2_score(y_test.iloc[:, i], preds[:, i]),
                            'RMSE': np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i])),
                            'MAE': mean_absolute_error(y_test.iloc[:, i], preds[:, i])
                        })
                        plot_actual_vs_pred(y_test.iloc[:, i], preds[:, i], title=f"{col}: Actual vs Predicted")

            elif task_choice == "Classification" and class_target:
                data = df[features_selected + [class_target]].dropna()
                X, y = data[features_selected], data[class_target].astype(str)
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42)
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                joblib.dump(pipe, model_path)
                st.success("✅ Classification model trained and saved!")
                st.metric("Accuracy", f"{acc:.4f}")
                st.text(classification_report(y_test, preds))
            else:
                st.error("Please select valid target column before training.")

# ---------------- Predict Tab ----------------
with tabs[3]:
    st.header("🔮 Prediction (Loads Saved Model if Available)")
    model_path = "saved_model.pkl"
    if os.path.exists(model_path):
        loaded_model = joblib.load(model_path)
        st.success("✅ Loaded saved model from disk.")
        feature_list = loaded_model.feature_names_in_.tolist() if hasattr(loaded_model, "feature_names_in_") else []
        if feature_list:
            st.subheader("Enter values for model features:")
            inputs = {}
            cols = st.columns(3)
            for i, f in enumerate(feature_list):
                with cols[i % 3]:
                    inputs[f] = st.text_input(f, "0", key=f"predict_{f}_{i}")
            if st.button("Predict Now"):
                try:
                    row = np.array([[float(inputs[f]) for f in feature_list]])
                    pred = loaded_model.predict(row)
                    if isinstance(pred[0], (np.ndarray, list)):
                        for i, val in enumerate(pred[0]):
                            st.metric(f"Predicted Output {i+1}", f"{val:.4f}")
                    else:
                        st.metric("Predicted Output", f"{pred[0]:.4f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.warning("Feature list not found. Retrain model.")
    else:
        st.info("No saved model found. Train one first.")

# ---------------- Alloy Type Helper ----------------
with tabs[4]:
    st.header("🧩 Alloy Type Helper")
    st.markdown("Enter alloy composition like `Fe:70, C:0.4, Cr:12, Ni:8`")
    text = st.text_area("Composition (key:value pairs or JSON)", height=120)
    elements = ['Fe','C','Cr','Ni','Al','Cu','Ti','Zn','Sn','Mn','Si']
    cols = st.columns(3)
    manual = {}
    for i, el in enumerate(elements):
        with cols[i % 3]:
            manual[el] = st.text_input(el, "", key=f"manual_{el}_{i}")
    if st.button("Analyze Alloy Type"):
        comp = {}
        if text:
            try:
                if text.strip().startswith("{"):
                    comp = json.loads(text)
                else:
                    for p in text.split(","):
                        if ":" in p:
                            k, v = p.split(":")
                            comp[k.strip()] = float(v.strip())
            except:
                st.warning("Parsing failed; using manual fields.")
        for k, v in manual.items():
            if v:
                try: comp[k] = float(v)
                except: pass
        alloy, reason = suggest_alloy_type(comp)
        st.success(alloy)
        st.write(reason)

st.caption("✅ App supports multi-target regression, classification, prediction, and alloy type analysis.")
