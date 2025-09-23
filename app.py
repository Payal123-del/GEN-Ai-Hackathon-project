import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re
import nltk
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# NLTK
nltk.download("stopwords", quiet=True)

# --------------------
# APP CONFIG
# --------------------
st.set_page_config(page_title="GenAIlytics: AI Job Market Analyzer", layout="wide")

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📊 GenAIlytics: AI Job Market Analyzer</h1>", unsafe_allow_html=True)
st.write("---")
# --------------------
# CLEAN DATA FUNCTION (पहले define करो)
# --------------------
@st.cache_data
def clean_data(df: pd.DataFrame):
    # ... आपका clean_data वाला पूरा code यहाँ ...
    return df


# --------------------
# LOAD DATA FUNCTION (clean_data के बाद)
# --------------------
@st.cache_data
def load_data():
    try:
        df_local = pd.read_csv("jobs_data.csv")
        st.sidebar.success("✅ Loaded local 'jobdata.csv'")
        return df_local
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Local file 'jobdata.csv' not found. Please upload CSV below.")

    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        st.sidebar.success("✅ Uploaded CSV loaded successfully!")
        return pd.read_csv(uploaded)

    st.error("❌ No dataset available. Please provide 'jobdata.csv' locally or upload a file.")
    return pd.DataFrame()


# --------------------
# LOAD + CLEAN
# --------------------
df = load_data()
if not df.empty:
    df = clean_data(df)
else:
    st.stop()





@st.cache_data
def clean_data(df: pd.DataFrame):
    df = df.copy()

    # 🔥 Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        if "job" in c and "title" in c:
            rename_map[c] = "job_title"
        if "company" in c and "location" in c:
            rename_map[c] = "company_location"
        if "salary" in c and "usd" in c:
            rename_map[c] = "salary_usd"
        if "experience" in c:
            rename_map[c] = "experience_level"
    df = df.rename(columns=rename_map)

    if "salary_usd" in df.columns:
        def to_float(x):
            try:
                return float(re.sub(r"[^0-9.]", "", str(x)))
            except:
                return np.nan
        df["salary_usd"] = df["salary_usd"].apply(to_float)

    def parse_exp(x):
        try:
            s = str(x).lower()
            m = re.search(r"(\d+)", s)
            if m: return float(m.group(1))
            if "senior" in s: return 7.0
            if "mid" in s: return 3.0
            if "entry" in s or "junior" in s: return 1.0
        except:
            return np.nan
        return np.nan

    if "experience_level" in df.columns:
        df["experience_yrs"] = df["experience_level"].apply(parse_exp)

    if "job_title" in df.columns:
        df["job_title"] = df["job_title"].fillna("Unknown")
    if "company_location" in df.columns:
        df["company_location"] = df["company_location"].fillna("Unknown")

    if "salary_usd" in df.columns and df["salary_usd"].notna().any():
        hi = df["salary_usd"].quantile(0.995)
        df = df[df["salary_usd"] <= hi]

    df.reset_index(drop=True, inplace=True)
    return df

# --------------------
st.sidebar.header("🔎 Filters")

job_filter = st.sidebar.multiselect(
    "Select Job Title",
    options=df["job_title"].unique() if "job_title" in df.columns else [],
    default=None,
)

loc_filter = st.sidebar.multiselect(
    "Select Location",
    options=df["company_location"].unique() if "company_location" in df.columns else [],
    default=None,
)

salary_min, salary_max = st.sidebar.slider(
    "Salary Range (USD)",
    min_value=int(df["salary_usd"].min()) if "salary_usd" in df.columns else 0,
    max_value=int(df["salary_usd"].max()) if "salary_usd" in df.columns else 100000,
    value=(
        int(df["salary_usd"].min()) if "salary_usd" in df.columns else 0,
        int(df["salary_usd"].max()) if "salary_usd" in df.columns else 100000,
    ),
)

# Apply filters
df_filtered = df.copy()
if job_filter:
    df_filtered = df_filtered[df_filtered["job_title"].isin(job_filter)]
if loc_filter:
    df_filtered = df_filtered[df_filtered["company_location"].isin(loc_filter)]
if "salary_usd" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["salary_usd"] >= salary_min) & (df_filtered["salary_usd"] <= salary_max)
    ]

# --------------------
# KPIs
# --------------------
st.subheader("📌 Descriptive Statistics & KPIs")
if not df_filtered.empty and "salary_usd" in df_filtered.columns:
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Salary (USD)", f"${df_filtered['salary_usd'].mean():,.0f}")
    c2.metric("Max Salary (USD)", f"${df_filtered['salary_usd'].max():,.0f}")
    c3.metric("Dataset Size", f"{len(df_filtered):,} records")

    st.dataframe(df_filtered.describe(include="all"))

st.write("---")

# --------------------
# VISUAL ANALYSIS
# --------------------
st.subheader("📊 Analysis (Filtered)")

if "salary_usd" in df_filtered.columns and "experience_yrs" in df_filtered.columns:
    fig = px.scatter(df_filtered, x="experience_yrs", y="salary_usd",
                     title="Experience vs Salary", opacity=0.6,
                     trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

if "job_title" in df_filtered.columns:
    top_jobs = df_filtered["job_title"].value_counts().head(10)
    fig2 = px.bar(top_jobs, title="Top Job Titles")
    st.plotly_chart(fig2, use_container_width=True)

st.write("---")

# --------------------
# PREDICTION MODEL
# --------------------
st.subheader("🤖 Salary Prediction")

if "salary_usd" in df_filtered.columns and "experience_yrs" in df_filtered.columns:
    X = df_filtered[["experience_yrs"]].fillna(0)
    y = df_filtered["salary_usd"]

    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write(f"**R² Score:** {r2_score(y_test, preds):.2f} | **MAE:** {mean_absolute_error(y_test, preds):.0f}")

        exp_input = st.slider("Select Experience (Years)", 0, 20, 3)
        pred_salary = model.predict([[exp_input]])[0]
        st.success(f"💰 Predicted Salary for {exp_input} years: **${pred_salary:,.0f} USD**")

st.write("---")

# --------------------
# DIAGNOSIS
# --------------------
st.subheader("🔎 Diagnosis (Factors Affecting Salary)")

if "salary_usd" in df_filtered.columns:
    corr = df_filtered.corr(numeric_only=True)
    if not corr.empty:
        fig3 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig3, use_container_width=True)

    st.write("✅ **Observation:** Salary is positively correlated with experience. Other factors may vary based on dataset.")

st.write("---")

# --------------------
# FINAL CONCLUSION
# --------------------
st.subheader("📌 Final Conclusion – Global AI Job Market & Salary Trends 2025")
st.markdown("""
<li><b>Rapid Growth:</b> AI job demand continues to grow worldwide at a record pace.</li>
<li><b>High Salaries:</b> AI roles offer premium pay, increasing with experience and specialization.</li>
<li><b>Key Drivers:</b> Experience, job level, skills, and location strongly influence salaries.</li>
<li><b>Remote & Hybrid Work:</b> Flexible work options are rising but still limited in some regions.</li>
<li><b>Future Outlook:</b> AI talent shortage ensures salaries and opportunities will keep expanding.</li>
</ul>
""", unsafe_allow_html=True)
