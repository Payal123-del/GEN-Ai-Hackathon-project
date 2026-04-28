import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

nltk.download("stopwords", quiet=True)

# --------------------
# APP CONFIG
# --------------------
st.set_page_config(page_title="GenAIlytics: AI Job Market Analyzer", layout="wide")

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📊 GenAIlytics: AI Job Market Analyzer</h1>", unsafe_allow_html=True)
st.write("---")

# --------------------
# CLEAN DATA
# --------------------
@st.cache_data
def clean_data(df: pd.DataFrame):
    df = df.copy()

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
            if m:
                return float(m.group(1))
            if "senior" in s:
                return 7.0
            if "mid" in s:
                return 3.0
            if "entry" in s or "junior" in s:
                return 1.0
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
# LOAD DATA
# --------------------
@st.cache_data
def load_data():
    try:
        df_local = pd.read_csv("jobs_data.csv")
        st.sidebar.success("✅ Loaded local CSV")
        return df_local
    except:
        st.sidebar.warning("Upload CSV file")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        return pd.read_csv(uploaded)

    st.stop()


df = load_data()
df = clean_data(df)

# --------------------
# FILTERS
# --------------------
st.sidebar.header("🔎 Filters")

job_filter = st.sidebar.multiselect(
    "Job Title",
    options=df["job_title"].unique() if "job_title" in df.columns else []
)

loc_filter = st.sidebar.multiselect(
    "Location",
    options=df["company_location"].unique() if "company_location" in df.columns else []
)

salary_min, salary_max = st.sidebar.slider(
    "Salary Range",
    int(df["salary_usd"].min()) if "salary_usd" in df.columns else 0,
    int(df["salary_usd"].max()) if "salary_usd" in df.columns else 100000,
    (
        int(df["salary_usd"].min()) if "salary_usd" in df.columns else 0,
        int(df["salary_usd"].max()) if "salary_usd" in df.columns else 100000
    )
)

df_filtered = df.copy()

if job_filter:
    df_filtered = df_filtered[df_filtered["job_title"].isin(job_filter)]

if loc_filter:
    df_filtered = df_filtered[df_filtered["company_location"].isin(loc_filter)]

if "salary_usd" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["salary_usd"] >= salary_min) &
        (df_filtered["salary_usd"] <= salary_max)
    ]

# --------------------
# KPIs
# --------------------
st.subheader("📌 KPIs")

if not df_filtered.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Salary", f"${df_filtered['salary_usd'].mean():,.0f}")
    c2.metric("Max Salary", f"${df_filtered['salary_usd'].max():,.0f}")
    c3.metric("Records", len(df_filtered))

# --------------------
# RECOMMENDATION
# --------------------
st.write("---")
st.subheader("🎯 Job Recommendation")

user_exp = st.slider("Your Experience", 0, 20, 2)

rec_jobs = df[
    (df["experience_yrs"] >= user_exp - 1) &
    (df["experience_yrs"] <= user_exp + 2)
]["job_title"].value_counts().head(5)

st.write("Suggested roles:", rec_jobs.index.tolist())

# --------------------
# VISUALS
# --------------------
st.write("---")
st.subheader("📊 Analysis")

fig = px.scatter(df_filtered, x="experience_yrs", y="salary_usd", trendline="ols")
st.plotly_chart(fig, use_container_width=True)

top_jobs = df_filtered["job_title"].value_counts().head(10)
st.plotly_chart(px.bar(top_jobs), use_container_width=True)

# --------------------
# ML MODEL
# --------------------
st.write("---")
st.subheader("🤖 Salary Prediction")

df_model = pd.get_dummies(df_filtered, columns=["job_title", "company_location"], drop_first=True)

X = df_model.drop("salary_usd", axis=1).fillna(0)
y = df_model["salary_usd"]

if len(X) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.write("R2:", round(r2_score(y_test, preds), 2))
    st.write("MAE:", int(mean_absolute_error(y_test, preds)))

    exp_input = st.slider("Experience for prediction", 0, 20, 3)

    input_df = pd.DataFrame([[exp_input]], columns=["experience_yrs"])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]

    pred = model.predict(input_df)[0]

    st.success(f"Predicted Salary: ${pred:,.0f}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    st.bar_chart(imp)

# --------------------
# INSIGHTS
# --------------------
st.write("---")
st.subheader("🤖 Quick Insights")

if st.button("Generate Insights"):
    st.success(f"Top Role: {df['job_title'].value_counts().idxmax()}")
    st.success(f"Best Location: {df.groupby('company_location')['salary_usd'].mean().idxmax()}")
    st.success(f"Experience Impact: {df[['experience_yrs','salary_usd']].corr().iloc[0,1]:.2f}")
