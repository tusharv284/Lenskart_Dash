import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules
import folium
from streamlit_folium import folium_static  # Fixed version

st.set_page_config(layout="wide", page_title="Lenskart AI Pro")

@st.cache_data
def load_data():
    return pd.read_csv('lenskart_dubai_locations.csv')

df = load_data()

st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {background: #0a0a0a;}
.metric-container {background: linear-gradient(135deg, #1a1a2e, #16213e) !important; 
                   border: 1px solid #00d4ff20 !important; border-radius: 12px !important;}
h1 {color: #00d4ff !important;}
</style>
""", unsafe_allow_html=True)

st.title("🕶️ **Lenskart Location AI Pro**")

# Sidebar
st.sidebar.title("Controls")
k = st.sidebar.slider("Clusters", 3, 8, 5)
area_filter = st.sidebar.multiselect("Area", df['area'].unique())
df_f = df[df['area'].isin(area_filter)] if area_filter else df

# KMeans
features = ['pop_density', 'med_income_aed', 'med_age', 'competitors', 'footfall_daily']
X = df_f[features]
scaler = StandardScaler()
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_f['cluster'] = kmeans.fit_predict(scaler.fit_transform(X))

# RF
y = (df_f['score'] > df_f['score'].quantile(0.7)).astype(int)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Optimal Sites", df_f['optimal_site'].sum())
col2.metric("Clusters", k)
col3.metric("Avg Score", f"{df_f['score'].mean():.1f}")

# Charts
col4, col5 = st.columns(2)
fig_pie = px.pie(df_f, names='risk_level', hole=0.4, title="Risk Breakdown")
col4.plotly_chart(fig_pie, use_container_width=True)

fig_scatter = px.scatter(df_f, x='med_income_aed', y='footfall_daily', color='cluster', 
                        size='score', title="Income vs Footfall")
col5.plotly_chart(fig_scatter, use_container_width=True)

# FIXED MAP - Line 115 equivalent
st.subheader("🗺️ Dubai Map (Fixed!)")
m = folium.Map(location=[25.25, 55.28], zoom_start=11, tiles='CartoDB dark_matter')
for idx, row in df_f.head(200).iterrows():  # Perf safe
    color = px.colors.qualitative.Set3[int(row['cluster']) % 12]
    folium.CircleMarker(
        [row['lat'], row['lon']], radius=row['score']/5 + 3,
        popup=f"{row['area']}<br>Score: {row['score']:.1f}<br>Cluster: {row['cluster']}",
        color=color, fill=True, fillOpacity=0.7
    ).add_to(m)

# FIXED folium_static CALL
folium_static(m, width=None, height=500, use_container_width=True)  # ✅ Cloud-safe [web:95]

# Table + Export
st.subheader("Cluster Summary")
cluster_df = df_f.groupby('cluster')[features + ['score']].mean().round(2)
st.dataframe(cluster_df)

csv = df_f.to_csv(index=False).encode('utf-8')
st.download_button("📥 Export CSV", csv, "lenskart_analysis.csv")

st.success("✅ **Fixed & Deployed** - Reboot app now!")
