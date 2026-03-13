import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import folium_static

st.set_page_config(layout="wide", page_title="Lenskart AI", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    return pd.read_csv('lenskart_dubai_locations.csv')

df = load_data()

# Neon CSS (single line - no truncation risk)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {background-color: #0a0a0a;}
.metric-container {background: linear-gradient(135deg, #1a1a2e, #16213e) !important; 
                   border: 1px solid #00d4ff20 !important; border-radius: 12px !important;}
.stPlotlyChart {border-radius: 12px !important; box-shadow: 0 8px 32px rgba(0,212,255,0.2) !important;}
h1 {color: #00d4ff !important; font-family: 'Roboto', sans-serif !important;}
</style>
""", unsafe_allow_html=True)

st.title("🕶️ Lenskart Location AI | Dubai")

# Sidebar
st.sidebar.title("Filters")
area_filter = st.sidebar.multiselect("Area", df['area'].unique())
df_f = df[df['area'].isin(area_filter)] if area_filter else df
k = st.sidebar.slider("Clusters", 3, 7, 5)

# Clustering
X = df_f[['pop_density', 'med_income_aed', 'med_age', 'competitors', 'footfall_daily']]
scaler = StandardScaler()
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_f['cluster'] = kmeans.fit_predict(scaler.fit_transform(X))

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Optimal Sites", df_f['optimal_site'].sum())
col2.metric("Clusters", k)
col3.metric("High Risk", (df_f['risk_level'] == 'High').sum())
col4.metric("Avg Score", f"{df_f['score'].mean():.1f}")

# Charts Row 1
col5, col6 = st.columns(2)
fig1 = px.pie(df_f, names='risk_level', hole=0.4, title="Risk Breakdown")
col5.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(df_f, x='med_income_aed', y='footfall_daily', 
                  color='cluster', size='score', title="Income vs Footfall")
col6.plotly_chart(fig2, use_container_width=True)

# Map
st.subheader("🗺️ Dubai Map")
m = folium.Map([25.25, 55.28], zoom_start=11, tiles='CartoDB dark_matter')
for _, row in df_f.head(100).iterrows():
    folium.CircleMarker([row['lat'], row['lon']], radius=5, 
                       color=['#ff0000','green','blue','orange'][int(row['cluster']%4)],
                       popup=f"Score: {row['score']:.1f}").add_to(m)
folium_static(m)

# Table
st.subheader("Cluster Summary")
cluster_stats = df_f.groupby('cluster')[['score', 'optimal_site']].mean().round(2)
st.dataframe(cluster_stats)

st.caption("✅ Fixed | Deploy-ready | Your CSV: lenskart_dubai_locations.csv")
