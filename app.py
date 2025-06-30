import streamlit as st
import plotly.graph_objects as go
import csv
import requests

st.set_page_config(page_title="AI Job Market", layout="wide")

# ---- Custom Styles ----
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #f0f4f8 0%, #e8f0fe 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #1e293b;
}

.job-button {
    background-color: #dbeafe;
    color: #1e3a8a;
    border: none;
    border-radius: 20px;
    padding: 8px 18px;
    margin: 6px;
    font-weight: 500;
    cursor: pointer;
    display: inline-block;
    text-align: center;
    transition: 0.3s;
}

.job-button:hover {
    background-color: #bfdbfe;
    transform: scale(1.05);
}

.centered {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 10px;
}

h3 {
    text-align: center;
    color: #1e3a8a;
    margin-top: 30px;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🌐 Global AI Job Market & Salary Trends – 2025")

# ---- Load Data ----
def read_csv_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['salary_usd'] = float(row['salary_usd'])
                data.append(row)
            except:
                continue
    return data

url="https://raw.githubusercontent.com/Payal123-del/GEN-Ai-Hackathon-project/refs/heads/main/jobs_data.csv"
def read_csv_from_github(raw_url):
    response = requests.get(raw_url)
    response.encoding = 'utf-8'
    lines = response.text.splitlines()
    reader = csv.DictReader(lines)
    data = []
    for row in reader:
        try:
            row['salary_usd'] = float(row['salary_usd'])
            data.append(row)
        except:
            continue
    return data
data = read_csv_from_github(url)

# ---- Extract Unique Job Titles & Locations ----
job_titles = sorted(set(row['job_title'] for row in data))
countries = sorted(set(row['company_location'] for row in data))

# ---- Centered Clickable Buttons ----
st.markdown("<h3>🧭 Click a Job Title to View Details</h3>", unsafe_allow_html=True)

cols = st.columns(4)
clicked_job = None

for index, job in enumerate(job_titles):
    col = cols[index % 4]
    if col.button(job, key=job):
        clicked_job = job

# ---- Display Job Info ----
if clicked_job:
    st.markdown(f"<h3>📌 Job Details for: {clicked_job}</h3>", unsafe_allow_html=True)
    for i, row in enumerate([r for r in data if r['job_title'] == clicked_job]):
        st.markdown(f"**{i+1}.** ${row['salary_usd']} — {row['company_location']}")

# ---- Plotly Bar Chart ----
if data:
    fig = go.Figure()
    grouped = {}

    for row in data:
        country = row['company_location']
        if country not in grouped:
            grouped[country] = {'x': [], 'y': []}
        grouped[country]['x'].append(row['job_title'])
        grouped[country]['y'].append(row['salary_usd'])

    for country, values in grouped.items():
        fig.add_trace(go.Bar(
            x=values['x'],
            y=values['y'],
            name=country
        ))

    st.markdown("""
        <h3>💰 Salary by Job Title & Country</h3>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
