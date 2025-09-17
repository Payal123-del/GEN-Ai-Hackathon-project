# GenAIlytics: Interactive AI Job Market Analyzer
# Single-file Streamlit app integrating: data loading, cleaning, visualization,
# NLP trend extraction, salary prediction, diagnosis, and recommendations.
# Save as GenAIlytics_streamlit_app.py and run: streamlit run GenAIlytics_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# ---- App config ----
st.set_page_config(page_title="GenAIlytics: AI Job Market Analyzer", layout='wide')

# ---- Helper functions ----
@st.cache_data
def read_csv_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.encoding = 'utf-8'
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

@st.cache_data
def clean_data(df: pd.DataFrame):
    # Basic cleaning & normalization
    df = df.copy()
    # Standard column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Try to unify key columns
    rename_map = {}
    for c in df.columns:
        if 'job' in c and 'title' in c:
            rename_map[c] = 'job_title'
        if 'company' in c and 'location' in c:
            rename_map[c] = 'company_location'
        if 'salary' in c and 'usd' in c:
            rename_map[c] = 'salary_usd'
        if 'experience' in c:
            rename_map[c] = 'experience_level'
        if 'description' in c:
            rename_map[c] = 'job_description'
    df = df.rename(columns=rename_map)

    # Salary cleaning
    if 'salary_usd' in df.columns:
        def to_float(x):
            try:
                if pd.isna(x):
                    return np.nan
                return float(re.sub(r'[^0-9.]','',str(x)))
            except:
                return np.nan
        df['salary_usd'] = df['salary_usd'].apply(to_float)

    # Fill missing categorical fields
    if 'job_title' in df.columns:
        df['job_title'] = df['job_title'].fillna('Unknown').astype(str)
    if 'company_location' in df.columns:
        df['company_location'] = df['company_location'].fillna('Unknown').astype(str)

    # Experience parse to numeric years when possible
    if 'experience_level' in df.columns:
        def parse_exp(x):
            try:
                s = str(x)
                m = re.search(r"(\d+)", s)
                if m:
                    return float(m.group(1))
                # map common bands
                s = s.lower()
                if 'senior' in s:
                    return 7.0
                if 'mid' in s:
                    return 3.0
                if 'entry' in s or 'junior' in s:
                    return 1.0
            except:
                pass
            return np.nan
        df['experience_yrs'] = df['experience_level'].apply(parse_exp)
    else:
        df['experience_yrs'] = np.nan

    # Drop rows without salary or job_title
    if 'salary_usd' in df.columns:
        df = df.dropna(subset=['job_title'])
        # Keep rows with salary or will be used in NLP/visualization only
    # Basic outlier removal: remove top 0.5% extremely large salaries
    if 'salary_usd' in df.columns and df['salary_usd'].notna().any():
        hi = df['salary_usd'].quantile(0.995)
        df = df[df['salary_usd'] <= hi]

    df.reset_index(drop=True, inplace=True)

    return df

@st.cache_data
def extract_top_skills(df: pd.DataFrame, n_terms=30):
    text_col = None
    if 'job_description' in df.columns:
        text_col = 'job_description'
    else:
        text_col = 'job_title'
    corpus = df[text_col].astype(str).str.lower().tolist()
    # simple tokenization & count
    words = []
    for doc in corpus:
        tokens = re.findall(r"[a-zA-Z]+", doc)
        for t in tokens:
            if t in STOPWORDS or len(t) < 2:
                continue
            words.append(t)
    freq = pd.Series(words).value_counts().head(n_terms)
    return freq

@st.cache_data
def cluster_job_titles(df: pd.DataFrame, k=6):
    texts = df['job_title'].astype(str).tolist()
    tf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tf.fit_transform(texts)
    # safe k
    k = min(k, max(2, int(len(texts)/10)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels

@st.cache_resource
def train_salary_model(df: pd.DataFrame):
    # Train a simple model on rows with salary available
    model_info = {}
    df_model = df.dropna(subset=['salary_usd']).copy()
    if df_model.shape[0] < 50:
        return None  # not enough data

    # features: job_title (categorical), company_location (categorical), experience_yrs (numeric)
    X = df_model[['job_title', 'company_location', 'experience_yrs']].copy()
    y = df_model['salary_usd']

    # fill experience missing with median
    X['experience_yrs'] = X['experience_yrs'].fillna(X['experience_yrs'].median())

    cat_feats = ['job_title', 'company_location']
    num_feats = ['experience_yrs']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_feats),
    ], remainder='passthrough')

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    pipeline = Pipeline([('pre', preprocessor), ('model', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    model_info['pipeline'] = pipeline
    model_info['mae'] = mean_absolute_error(y_test, preds)
    model_info['r2'] = r2_score(y_test, preds)
    model_info['n_train'] = len(X_train)
    return model_info

# ---- Layout ----
st.title("GenAIlytics: Interactive AI Job Market Analyzer")
st.markdown("A single-file Streamlit app with Visualization, Analysis, Prediction, Diagnosis & Solutions.\n*Upload/Load dataset, explore, and get model-backed recommendations.*")

# Sidebar: Data source & options
st.sidebar.header('Data & Settings')
use_example = st.sidebar.checkbox('Load example dataset from GitHub (recommended)', value=True)

if use_example:
    url = st.sidebar.text_input('GitHub raw CSV URL', value='https://raw.githubusercontent.com/Payal123-del/GEN-Ai-Hackathon-project/main/jobs_data.csv')
    df_raw = read_csv_from_url(url)
else:
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.DataFrame()

st.sidebar.markdown('---')
if not df_raw.empty:
    df = clean_data(df_raw)
else:
    st.warning('No data loaded yet. Please upload or enable example dataset.')
    st.stop()

# Top-level KPIs
st.markdown('## Overview & Key Metrics')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total Records', int(len(df)))
with col2:
    if 'salary_usd' in df.columns and df['salary_usd'].notna().any():
        st.metric('Avg Salary (USD)', f"${df['salary_usd'].mean():.0f}")
    else:
        st.metric('Avg Salary (USD)', 'N/A')
with col3:
    st.metric('Unique Job Titles', int(df['job_title'].nunique()))
with col4:
    st.metric('Unique Countries', int(df['company_location'].nunique()))

st.markdown('---')

# Filters
st.markdown('## Filters & Quick Selection')
filter_col1, filter_col2, filter_col3 = st.columns([3,2,2])
with filter_col1:
    sel_job = st.selectbox('Select Job Title (or All)', options=['All'] + sorted(df['job_title'].unique().tolist()))
with filter_col2:
    sel_country = st.selectbox('Select Country (or All)', options=['All'] + sorted(df['company_location'].unique().tolist()))
with filter_col3:
    min_salary, max_salary = st.slider('Salary USD range',
                                       0, int(df['salary_usd'].max() if df['salary_usd'].notna().any() else 100000),
                                       (0, int(df['salary_usd'].max() if df['salary_usd'].notna().any() else 100000)))

filt = df.copy()
if sel_job != 'All':
    filt = filt[filt['job_title'] == sel_job]
if sel_country != 'All':
    filt = filt[filt['company_location'] == sel_country]
if 'salary_usd' in filt.columns and filt['salary_usd'].notna().any():
    filt = filt[(filt['salary_usd'] >= min_salary) & (filt['salary_usd'] <= max_salary)]

# Visualization: Salary distribution by job
st.markdown('## Visualizations')
viz_col1, viz_col2 = st.columns([2,1])
with viz_col1:
    st.markdown('### Salary by Job Title (grouped by country)')
    if not filt.empty and 'salary_usd' in filt.columns and filt['salary_usd'].notna().any():
        grouped = filt.groupby(['job_title','company_location'])['salary_usd'].median().reset_index()
        fig = px.bar(grouped, x='job_title', y='salary_usd', color='company_location', barmode='group', title='Median Salary by Job Title & Country')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No salary data available for the selected filters.')

with viz_col2:
    st.markdown('### Top Job Titles by Count')
    top_jobs = filt['job_title'].value_counts().head(10)
    fig2 = go.Figure(go.Bar(x=top_jobs.values, y=top_jobs.index, orientation='h'))
    fig2.update_layout(height=400, margin=dict(l=100))
    st.plotly_chart(fig2, use_container_width=True)

# NLP: Top skills / keywords
st.markdown('## NLP Insights: Trending Skills & Title Clusters')
skills = extract_top_skills(df, n_terms=30)
skill_col1, skill_col2 = st.columns([1,2])
with skill_col1:
    st.markdown('### Top Keywords (from job titles/descriptions)')
    st.table(skills.reset_index().rename(columns={'index':'term', 0:'count'}).head(20))
with skill_col2:
    st.markdown('### Job Title Clusters (KMeans on TF-IDF)')
    if len(df) > 10:
        labels = cluster_job_titles(df, k=6)
        df['jt_cluster'] = labels
        cluster_summary = df.groupby('jt_cluster')['job_title'].agg(lambda x: ', '.join(pd.unique(x)[:6])).reset_index()
        st.write(cluster_summary)
    else:
        st.info('Not enough data to perform clustering.')

st.markdown('---')

# Prediction section
st.markdown('## Salary Prediction (Model-backed)')
model_info = train_salary_model(df)
if model_info is None:
    st.info('Not enough labeled salary data to train a reliable prediction model. Add more salary entries and retry.')
else:
    st.markdown(f"Trained model on ~{model_info['n_train']} records. MAE=${model_info['mae']:.0f}, R2={model_info['r2']:.2f}")
    pred_col1, pred_col2 = st.columns([2,1])
    with pred_col1:
        st.markdown('### Try the Salary Predictor')
        job_input = st.selectbox('Job Title for prediction', options=sorted(df['job_title'].unique().tolist()))
        country_input = st.selectbox('Company Location for prediction', options=sorted(df['company_location'].unique().tolist()))
        exp_input = st.number_input('Years of experience', min_value=0.0, max_value=40.0, value=2.0)
        if st.button('Predict Salary'):
            pipeline = model_info['pipeline']
            X_new = pd.DataFrame([{'job_title': job_input, 'company_location': country_input, 'experience_yrs': exp_input}])
            pred = pipeline.predict(X_new)[0]
            st.success(f"Estimated Salary (USD): ${pred:,.0f}")
    with pred_col2:
        st.markdown('### Model Insights')
        st.write('Model type: RandomForestRegressor in a pipeline with OneHotEncoder.')
        st.write('Tip: If MAE is large, consider gathering more labeled salary data & feature engineering.')

st.markdown('---')

# Diagnosis & Solutions
st.markdown('## Diagnosis: Market Gaps & Suggested Solutions')
# Diagnosis examples
with st.expander('Diagnosis Summary (automated)'):
    # Top paying countries
    if 'salary_usd' in df.columns and df['salary_usd'].notna().any():
        country_salary = df.groupby('company_location')['salary_usd'].median().sort_values(ascending=False)
        st.write('Top 5 countries by median salary:')
        st.table(country_salary.head(5).reset_index().rename(columns={'company_location':'country','salary_usd':'median_salary'}))

    # Salary gap example
    if 'job_title' in df.columns and 'salary_usd' in df.columns:
        pivot = df.groupby(['job_title','company_location'])['salary_usd'].median().reset_index()
        # find job titles present in >=2 countries
        job_counts = pivot['job_title'].value_counts()
        multi_country_jobs = job_counts[job_counts>=2].index.tolist()
        gaps = []
        for job in multi_country_jobs:
            subset = pivot[pivot['job_title']==job].sort_values('salary_usd')
            if len(subset) >=2:
                low = subset.iloc[0]
                high = subset.iloc[-1]
                rel = (high['salary_usd'] - low['salary_usd']) / max(1, low['salary_usd'])
                gaps.append({'job_title':job, 'low_country':low['company_location'], 'low_salary':low['salary_usd'],
                             'high_country':high['company_location'], 'high_salary':high['salary_usd'], 'rel_gap':rel})
        gaps_df = pd.DataFrame(gaps).sort_values('rel_gap', ascending=False).head(5)
        if not gaps_df.empty:
            st.write('Top 5 cross-country salary gaps (sample):')
            st.table(gaps_df)
        else:
            st.info('No significant cross-country gaps found in the provided dataset.')

with st.expander('Actionable Solutions (for Job Seekers & Companies)'):
    st.markdown('**For Job Seekers:**')
    st.write('- Target upskilling in top keywords shown above (e.g., cloud, python, ml)')
    st.write('- Consider markets/countries where the role pays higher; remote-friendly companies may match those bands')
    st.markdown('**For Companies / HR:**')
    st.write('- Benchmark salaries for roles with large gaps to reduce attrition risk')
    st.write('- Improve job descriptions to include required skills discovered by NLP to attract right talent')

st.markdown('---')

# Simple GenAI-like Q&A (retrieval-based)
st.markdown('## GenAI Q&A Assistant (Data-backed)')
qa_col1, qa_col2 = st.columns([3,1])
with qa_col1:
    question = st.text_input('Ask a question about the dataset (e.g., "Which AI job pays most in India?")')
    if st.button('Ask') and question.strip() != '':
        q = question.lower()
        answered = False
        # Simple patterns
        if 'which' in q and 'pays' in q and 'most' in q:
            # try country mention
            m = re.search(r'in\s+([a-zA-Z ]+)', q)
            if m:
                country = m.group(1).strip().title()
                if country in df['company_location'].values:
                    sub = df[df['company_location'] == country]
                    if 'salary_usd' in sub.columns and sub['salary_usd'].notna().any():
                        top = sub.groupby('job_title')['salary_usd'].median().sort_values(ascending=False).head(1)
                        if not top.empty:
                            jt = top.index[0]
                            val = top.iloc[0]
                            st.write(f"**Answer:** {jt} — median salary ${val:,.0f} in {country} (from dataset)")
                            answered = True
        # Average salary question
        if not answered and ('average' in q or 'avg' in q) and 'salary' in q:
            m = re.search(r'in\s+([a-zA-Z ]+)', q)
            if m:
                country = m.group(1).strip().title()
                if country in df['company_location'].values and 'salary_usd' in df.columns:
                    val = df[df['company_location']==country]['salary_usd'].median()
                    st.write(f"**Answer:** Median salary in {country} is approximately ${val:,.0f}")
                    answered = True
        # Keyword search
        if not answered:
            # fallback: show top rows matching keywords
            key_terms = re.findall(r"[a-zA-Z]+", q)
            key_terms = [k for k in key_terms if k not in ('which','what','is','the','in','of','and','for')]
            subset = df.copy()
            for kt in key_terms[:3]:
                subset = subset[subset['job_title'].str.contains(kt, case=False, na=False) | subset.get('job_description','').astype(str).str.contains(kt, case=False, na=False)]
            if not subset.empty:
                st.write('**Answer (sample records matching query):**')
                st.dataframe(subset.head(10))
                answered = True
        if not answered:
            st.write("Sorry — I couldn't find a clear answer in the dataset. Try asking about a job title or a country present in the data.")
with qa_col2:
    st.markdown('**Tips for questions**')
    st.write('- Ask about specific countries or job titles present in dataset')
    st.write('- Examples: "Which AI job pays most in Canada?", "Average salary of Data Scientist in USA"')

st.markdown('---')

# Export & Download
st.markdown('## Export & Save')
export_col1, export_col2 = st.columns([2,1])
with export_col1:
    st.markdown('### Download current filtered view as CSV')
    if not filt.empty:
        csv = filt.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', csv, file_name='filtered_jobs.csv', mime='text/csv')
    else:
        st.info('No records to export for current filters.')
with export_col2:
    st.markdown('### Save model (pickle)')
    if model_info is not None:
        import pickle
        buf = io.BytesIO()
        pickle.dump(model_info['pipeline'], buf)
        buf.seek(0)
        st.download_button('Download model (pickle)', buf, file_name='salary_model.pkl', mime='application/octet-stream')
    else:
        st.info('No trained model available to download.')

st.markdown('---')

# Footer / How to explain in interview
with st.expander('How to explain this project in 6 lines (interview-ready)'):
    st.markdown("""
1. "I built **GenAIlytics**, an end-to-end interactive dashboard using Streamlit that analyzes the global AI job market."
2. "It ingests raw job datasets, performs cleaning, normalizes salaries and extracts features like experience and location."
3. "I used TF-IDF and KMeans to cluster job titles and extract trending skills via NLP." 
4. "For salary prediction I trained a RandomForest pipeline with OneHot encoding; I evaluate using MAE and R²."
5. "I implemented diagnostic checks (cross-country salary gaps, top skill shortages) and gave actionable recommendations."
6. "Finally, I added an interactive retrieval-based GenAI-like assistant for dataset Q&A and export features for stakeholders."""
)

st.success('Project ready — customize dataset, tweak model & features, and you have an interview-ready GenAI Analytics app!')



 

