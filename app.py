import os
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
# APP CONFIG & STYLING
# --------------------
st.set_page_config(page_title="GenAIlytics: AI Job Market Analyzer", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.main-header {
    text-align: center;
    color: #1B5E20;
    font-weight: 700;
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}

.sub-header {
    text-align: center;
    color: #64748B;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.section-title {
    color: #1E293B;
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #F1F5F9;
    padding-bottom: 0.4rem;
}

.card-container {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}

.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2E7D32;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.06);
}

.metric-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
}

.prediction-box {
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 1.05rem;
    color: #14532D;
    font-weight: 600;
}

.insight-box {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
}

.insight-item {
    padding: 0.6rem 0;
    font-size: 0.95rem;
    color: #334155;
    border-bottom: 1px dashed #E2E8F0;
}

.insight-item:last-child {
    border-bottom: none;
}

[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}

.status-box-success {
    background-color: #F0FDF4;
    border: 1px solid #86EFAC;
    color: #166534;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.status-box-warning {
    background-color: #FEFCE8;
    border: 1px solid #FDE047;
    color: #854D0E;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.status-box-error {
    background-color: #FEF2F2;
    border: 1px solid #FCA5A5;
    color: #991B1B;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# HEADER
# --------------------
st.markdown('<div class="main-header">GenAIlytics: AI Job Market Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven analytics, compensation benchmarks, and predictive modeling for AI roles</div>', unsafe_allow_html=True)

# --------------------
# DATA FUNCTIONS (CACHED - NO UI COMMANDS)
# --------------------
@st.cache_data
def clean_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # Specific column renaming without creating duplicate column names
    rename_map = {}
    for c in df.columns:
        if "job" in c and "title" in c:
            rename_map[c] = "job_title"
        elif "company" in c and "location" in c:
            rename_map[c] = "company_location"
        elif "salary" in c and "usd" in c:
            rename_map[c] = "salary_usd"
        elif c == "experience_level" or (c.startswith("exp") and "level" in c):
            rename_map[c] = "experience_level"
        elif c in ["years_experience", "years_exp", "experience_years", "experience_yrs"]:
            rename_map[c] = "experience_yrs"

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    # Salary cleaning
    if "salary_usd" in df.columns:
        df["salary_usd"] = df["salary_usd"].apply(lambda x: float(re.sub(r"[^0-9.]", "", str(x))) if pd.notna(x) else np.nan)
        df = df[df["salary_usd"] <= df["salary_usd"].quantile(0.995)]

    # Experience parsing
    def parse_exp(x):
        s = str(x).lower().strip()
        m = re.search(r"(\d+)", s)
        if m: return float(m.group(1))
        if "senior" in s or s == "se": return 7.0
        if "executive" in s or s == "ex": return 10.0
        if "mid" in s or s == "mi": return 3.0
        if "entry" in s or "junior" in s or s == "en": return 1.0
        return np.nan

    if "experience_yrs" not in df.columns and "experience_level" in df.columns:
        df["experience_yrs"] = df["experience_level"].apply(parse_exp)
    elif "experience_yrs" in df.columns:
        df["experience_yrs"] = pd.to_numeric(df["experience_yrs"], errors="coerce")
        if "experience_level" in df.columns:
            df["experience_yrs"] = df["experience_yrs"].fillna(df["experience_level"].apply(parse_exp))

    df["job_title"] = df.get("job_title", pd.Series(["Unknown"]*len(df))).fillna("Unknown")
    df["company_location"] = df.get("company_location", pd.Series(["Unknown"]*len(df))).fillna("Unknown")
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data
def load_data(file_source) -> pd.DataFrame:
    try:
        return pd.read_csv(file_source)
    except Exception:
        return pd.DataFrame()


# --------------------
# SIDEBAR DATA SOURCE & NOTIFICATIONS
# --------------------
st.sidebar.markdown('<h4 style="color: #1E293B; font-weight: 600; margin-bottom: 0.5rem;">Data Source</h4>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

raw_df = pd.DataFrame()
if uploaded is not None:
    raw_df = load_data(uploaded)
    if not raw_df.empty:
        st.sidebar.markdown('<div class="status-box-success">Uploaded dataset loaded successfully</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-box-error">Failed to read uploaded CSV</div>', unsafe_allow_html=True)
elif os.path.exists("jobs_data.csv"):
    raw_df = load_data("jobs_data.csv")
    if not raw_df.empty:
        st.sidebar.markdown('<div class="status-box-success">Loaded local dataset (jobs_data.csv)</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-box-warning">Local file found but could not be parsed</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-box-warning">Local dataset not found. Please upload a CSV file</div>', unsafe_allow_html=True)

if raw_df.empty:
    st.error("No dataset provided. Please upload a CSV file or ensure 'jobs_data.csv' is present.")
    st.stop()

df = clean_data(raw_df)

# --------------------
# SIDEBAR FILTERS
# --------------------
st.sidebar.markdown('<h4 style="color: #1E293B; font-weight: 600; margin-top: 1.25rem; margin-bottom: 0.5rem;">Filters</h4>', unsafe_allow_html=True)
job_filter = st.sidebar.multiselect("Job Title", df["job_title"].unique(), default=None)
loc_filter = st.sidebar.multiselect("Location", df["company_location"].unique(), default=None)
salary_min, salary_max = st.sidebar.slider(
    "Salary Range (USD)",
    min_value=int(df["salary_usd"].min()) if "salary_usd" in df.columns else 0,
    max_value=int(df["salary_usd"].max()) if "salary_usd" in df.columns else 100000,
    value=(int(df["salary_usd"].min()), int(df["salary_usd"].max()))
)

# Apply filters
df_filtered = df.copy()
if job_filter: df_filtered = df_filtered[df_filtered["job_title"].isin(job_filter)]
if loc_filter: df_filtered = df_filtered[df_filtered["company_location"].isin(loc_filter)]
if "salary_usd" in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered["salary_usd"] >= salary_min) & (df_filtered["salary_usd"] <= salary_max)]

# --------------------
# DESCRIPTIVE KPIs
# --------------------
st.markdown('<div class="section-title">Key Stats & KPIs</div>', unsafe_allow_html=True)

if not df_filtered.empty:
    if "salary_usd" in df_filtered.columns:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Average Salary</div>
                    <div class="metric-value">${df_filtered['salary_usd'].mean():,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Max Salary</div>
                    <div class="metric-value">${df_filtered['salary_usd'].max():,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Records</div>
                    <div class="metric-value">{len(df_filtered):,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    st.dataframe(df_filtered.describe(include="all"), use_container_width=True)

# --------------------
# VISUAL ANALYSIS
# --------------------
st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)

if "salary_usd" in df_filtered.columns and "experience_yrs" in df_filtered.columns:
    try:
        fig = px.scatter(
            df_filtered,
            x="experience_yrs",
            y="salary_usd",
            title="Experience vs Salary",
            labels={"experience_yrs": "Experience (Years)", "salary_usd": "Salary (USD)"},
            opacity=0.6,
            trendline="ols",
            color_discrete_sequence=["#2E7D32"]
        )
    except Exception:
        fig = px.scatter(
            df_filtered,
            x="experience_yrs",
            y="salary_usd",
            title="Experience vs Salary",
            labels={"experience_yrs": "Experience (Years)", "salary_usd": "Salary (USD)"},
            opacity=0.6,
            color_discrete_sequence=["#2E7D32"]
        )
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

if "job_title" in df_filtered.columns:
    top_jobs = df_filtered["job_title"].value_counts().head(10).reset_index()
    top_jobs.columns = ["job_title", "count"]
    fig2 = px.bar(
        top_jobs,
        x="count",
        y="job_title",
        orientation="h",
        title="Top Job Titles by Openings",
        labels={"count": "Number of Openings", "job_title": ""},
        color_discrete_sequence=["#2E7D32"]
    )
    fig2.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------
# SALARY PREDICTION
# --------------------
st.markdown('<div class="section-title">Salary Prediction</div>', unsafe_allow_html=True)

if "salary_usd" in df_filtered.columns and "experience_yrs" in df_filtered.columns and len(df_filtered) > 10:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    X = df_filtered[["experience_yrs"]].fillna(0)
    y = df_filtered["salary_usd"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    st.markdown(f"**Model Performance:** R² Score: `{r2_score(y_test, preds):.2f}` | MAE: `${mean_absolute_error(y_test, preds):,.0f}`")
    
    exp_input = st.slider("Experience (Years)", 0, 20, 3)
    predicted_val = model.predict([[exp_input]])[0]
    
    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted Salary: <strong>${predicted_val:,.0f} USD</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------
# CORRELATION HEATMAP
# --------------------
st.markdown('<div class="section-title">Correlation Analysis</div>', unsafe_allow_html=True)

if "salary_usd" in df_filtered.columns:
    corr = df_filtered.corr(numeric_only=True)
    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="Greens",
        aspect="auto"
    )
    fig3.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("Salary positively correlates with experience. Other factors vary by dataset.")

# --------------------
# FINAL DATA-DRIVEN INSIGHTS
# --------------------
st.markdown('<div class="section-title">Data-Driven Insights – AI Job Market 2025</div>', unsafe_allow_html=True)

if not df_filtered.empty:
    # 1️⃣ Top Locations by Number of Jobs
    if "company_location" in df_filtered.columns:
        top_locs = df_filtered["company_location"].value_counts().head(10).reset_index()
        top_locs.columns = ["location", "count"]
        fig_loc = px.bar(
            top_locs,
            x="location",
            y="count",
            title="Top Locations by Job Openings",
            labels={"location": "Location", "count": "Job Openings"},
            color_discrete_sequence=["#388E3C"]
        )
        fig_loc.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=50, b=40),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_loc, use_container_width=True)

    # 2️⃣ Average Salary by Experience Level
    if "experience_yrs" in df_filtered.columns and "salary_usd" in df_filtered.columns:
        avg_salary_exp = df_filtered.groupby("experience_yrs")["salary_usd"].mean().reset_index()
        fig_exp = px.line(
            avg_salary_exp,
            x="experience_yrs",
            y="salary_usd",
            title="Average Salary vs Experience (Years)",
            labels={"experience_yrs": "Experience (Years)", "salary_usd": "Average Salary (USD)"},
            markers=True,
            color_discrete_sequence=["#1B5E20"]
        )
        fig_exp.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=50, b=40),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    # 3️⃣ Top Job Titles by Salary
    if "job_title" in df_filtered.columns and "salary_usd" in df_filtered.columns:
        top_salary_jobs = df_filtered.groupby("job_title")["salary_usd"].mean().sort_values(ascending=False).head(10).reset_index()
        top_salary_jobs.columns = ["job_title", "avg_salary"]
        fig_jobs = px.bar(
            top_salary_jobs,
            x="avg_salary",
            y="job_title",
            orientation="h",
            title="Top 10 Job Titles by Average Salary",
            labels={"avg_salary": "Average Salary (USD)", "job_title": ""},
            color_discrete_sequence=["#2E7D32"]
        )
        fig_jobs.update_layout(
            template="plotly_white",
            height=450,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=40, r=40, t=50, b=40),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_jobs, use_container_width=True)

    # 4️⃣ Key Observations (Dynamic)
    st.markdown('<div class="section-title" style="font-size: 1.1rem; border-bottom: none; margin-top: 1rem; margin-bottom: 0.25rem;">Key Observations</div>', unsafe_allow_html=True)
    obs = []
    # Highest paying location
    if "company_location" in df_filtered.columns and "salary_usd" in df_filtered.columns:
        best_loc = df_filtered.groupby("company_location")["salary_usd"].mean().idxmax()
        obs.append(f"Highest average salary location: <strong>{best_loc}</strong>")
    
    # Experience impact
    if "experience_yrs" in df_filtered.columns and "salary_usd" in df_filtered.columns:
        corr_val = df_filtered[["experience_yrs", "salary_usd"]].corr().iloc[0,1]
        obs.append(f"Salary correlation with experience: <strong>{corr_val:.2f}</strong> (positive correlation)")

    # Job titles with high salaries
    if "job_title" in df_filtered.columns and "salary_usd" in df_filtered.columns:
        high_salary_job = df_filtered.groupby("job_title")["salary_usd"].mean().idxmax()
        obs.append(f"Top paying role: <strong>{high_salary_job}</strong>")

    obs_html = '<div class="insight-box">'
    for idx, item in enumerate(obs, 1):
        obs_html += f'<div class="insight-item"><strong>{idx}.</strong> {item}</div>'
    obs_html += '</div>'
    st.markdown(obs_html, unsafe_allow_html=True)

else:
    st.info("No data available for generating insights. Please adjust filters or upload a dataset.")
