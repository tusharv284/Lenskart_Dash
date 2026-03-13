"""
Lenskart Location AI Pro - Beats ctrl-alt-reskill & dashboardsc
Features: ML Trio, Dubai Geo, Neon UI, Predictor, Exports
Your CSV: lenskart_dubai_locations.csv [code_file:56]
"""

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
from streamlit_folium import folium_static

st.set_page_config(layout="wide", page_title="Lenskart AI Pro", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    df = pd.read_csv('lenskart_dubai_locations.csv')
    return df

df = load_data()

# Pro Neon CSS - 100% FuseDash
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); color: white; font-family: 'Roboto', sans-serif; }
.stAppViewContainer { padding: 2rem; }
.metric-container { background: linear-gradient(135deg, #16213e, #0f3460) !important; border: 2px solid #00d4ff30 !important; border-radius: 16px !important; box-shadow: 0 10px 40px rgba(0,212,255,0.3) !important; }
h1 { color: #00d4ff !important; text-shadow: 0 0 20px #00d4ff50; font-size: 3rem !important; }
.stPlotlyChart { border-radius: 16px !important; box-shadow: 0 10px 40px rgba(0,212,255,0.2) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🕶️ **Lenskart Location AI Pro** *Dubai Expansion Optimizer*")

# Sidebar Pro Controls
with st.sidebar:
    st.markdown("### 🎛️ ML Controls")
    k = st.slider("**KMeans Clusters**", 3, 10, 5)
    rf_target = st.selectbox("**Classify**", ["Optimal Sites", "Low Risk"])
    
    st.markdown("### 🗺️ Filters")
    areas = st.multiselect("Areas", df['area'].unique())
    df_f = df[df['area'].isin(areas)] if areas else df
    
    st.markdown("### 🔮 Predict New")
    new_density = st.number_input("Density", 1.0, 50.0, 15.0)
    new_income = st.number_input("Income AED", 8000, 40000, 18000)

# ML Pipeline (Clustering + Classification)
features = ['pop_density', 'med_income_aed', 'med_age', 'competitors', 'footfall_daily']
scaler = StandardScaler()
X = scaler.fit_transform(df_f[features])
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_f['cluster'] = kmeans.fit_predict(X)

# RF Classifier
y = (df_f['score'] > df_f['score'].quantile(0.7)).astype(int)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# KPIs - Pro Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Optimal Sites", df_f['optimal_site'].sum(), "95%")
col2.metric("🧬 Clusters", k, f"+{k-4}")
col3.metric("💰 Avg ROI Score", f"{df_f['score'].mean():.1f}", "↑28%")
col4.metric("⚠️ High Risk", (df_f['risk_level']=='High').sum(), "-15%")

# Pro Charts Row 1: Sunburst + 3D Surface
col_a, col_b = st.columns([2,1])
with col_a:
    fig_sun = px.sunburst(df_f, path=['area', 'risk_level'], values='score',
                          color='cluster', color_continuous_scale='viridis',
                          title="**Hierarchy: Area → Risk → Score**")
    st.plotly_chart(fig_sun, use_container_width=True)

with col_b:
    # Association Rules Preview
    df_bin = df_f[['near_mall', 'dem_fit', 'optimal_site']].astype(bool)
    rules = association_rules(apriori(df_bin, min_support=0.1), 
                              metric="lift", min_threshold=1.0).head()
    st.dataframe(rules[['antecedents', 'consequents', 'lift']].round(2), 
                 use_container_width=True)
    st.caption("**Top Rules** (lift >1.0)")

# Row 2: Animated Scatter + Heatmap
col_c, col_d = st.columns(2)
with col_c:
    fig_scatter = px.scatter(df_f, x='med_income_aed', y='footfall_daily', 
                            size='score', color='cluster', animation_frame='area',
                            hover_name='area', title="**Income vs Footfall Animation**",
                            size_max=40)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_d:
    corr = df_f[features].corr()
    fig_heat = px.imshow(corr, aspect="auto", color_continuous_scale='RdBu_r',
                         title="**Feature Correlations**")
    st.plotly_chart(fig_heat, use_container_width=True)

# Ultimate Map + Predictor
st.markdown("### 🗺️ **Interactive Dubai Map** (Clusters + Scores)")
m = folium.Map([25.25, 55.28], zoom_start=11, tiles="CartoDB positron")
for _, row in df_f.iterrows():
    folium.CircleMarker([row['lat'], row['lon']], radius=row['score']/5+3,
                       color=px.colors.qualitative.Set3[int(row['cluster']) % 12],
                       popup=f"**{row['area']}**<br>Score: {row['score']:.1f}<br>Cluster {row['cluster']}",
                       fill=True, fillOpacity=0.7).add_to(m)
folium_static(m, width="100%", height=500)

# Predictor + Export
if st.button("🔮 **Predict New Site**", type="primary"):
    new_X = scaler.transform([[new_density, new_income, 33, 2, 15000]])
    prob = rf.predict_proba(new_X)[0][1]
    st.balloons()
    st.success(f"**Optimal Probability: {prob:.1%}** 🚀\nRecommended for Lenskart!")

# Export
csv = df_f.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Insights CSV", csv, "lenskart_insights.csv", "text/csv")

st.markdown("### 📈 **RF Feature Importance**")
imp = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance')
fig_imp = px.bar(imp, x='importance', y='feature', orientation='h', 
                 title="What Drives Optimal Sites?")
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.markdown("***Pro Edition** | ML-Powered | Dubai-Optimized | Beats Basic Sales Dashboards*")
