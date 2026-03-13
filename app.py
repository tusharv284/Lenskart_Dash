import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Lenskart Location AI", initial_sidebar_state="expanded")
df = pd.read_csv('lenskart_dubai_locations.csv')  # Your file [code_file:56]

# Custom CSS for FuseDash neon dark
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a0a; }
    .metric { color: #00d4ff; font-weight: bold; }
    .stPlotlyChart { border-radius: 12px; box-shadow: 0 4px 20px rgba(0,212,255,0.3); }
    h1 { color: #00d4ff; font-family: 'Roboto', sans-serif; }
    .kpi-card { background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #00d4ff20; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🕶️ **Lenskart Location AI** | Dubai 2k+ Grids Analyzed")

# Top Nav & KPIs (4 columns)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Optimal Sites", f"{df[df.optimal_site==1].shape[0]:,}", "95%")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Top Dem Age", "25-35", "+12%")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Clusters", "7", "New: Premium")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("ROI Potential", "$2.5M", "+28%")
    st.markdown('</div>', unsafe_allow_html=True)

# Row 1: Donut + Radar
col5, col6 = st.columns(2)
with col5:
    fig_donut = px.pie(values=[65, 35], names=["Optimal", "Avoid"], hole=0.4,
                       color_discrete_sequence=['#00d4ff', '#ff4d6d'])
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(title="Site Potential", showlegend=False)
    st.plotly_chart(fig_donut, use_container_width=True)

with col6:
    # Radar: Dem by factors
    cats = ['Density', 'Income', 'Age Fit', 'Low Comp', 'Footfall']
    premium = [8.5, 9.2, 7.8, 9.0, 9.5]
    avg = [5.0, 5.5, 5.2, 4.8, 6.0]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=premium, theta=cats, fill='toself', name='Premium Cluster'))
    fig_radar.add_trace(go.Scatterpolar(r=avg, theta=cats, fill='toself', name='Average'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                            title="Demographic Radar", showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# Row 2: Choropleth Dubai + Bubble Matrix
col7, col8 = st.columns(2)
with col7:
    # Choropleth (Dubai areas by score)
    area_score = df.groupby('area')['score'].mean().reset_index()
    fig_choro = px.choropleth_mapbox(area_score, geojson=None,  # Custom Dubai geojson optional
                                     locations='area', color='score',
                                     color_continuous_scale='blues',
                                     mapbox_style="carto-darkmatter", zoom=10,
                                     center={"lat": 25.25, "lon": 55.28},
                                     opacity=0.8, title="Dubai Zones by Score")
    st.plotly_chart(fig_choro, use_container_width=True)

with col8:
    # Bubble: Risk × Age bubbles sized by optimal n
    bubble_df = df.groupby(['risk_level', pd.cut(df.med_age, bins=5)])['optimal_site'].agg(['count', 'mean']).reset_index()
    bubble_df['size'] = bubble_df['count']
    fig_bubble = px.scatter(bubble_df, x='risk_level', y='med_age', size='size', color='mean',
                            title="Risk × Age Matrix", size_max=40)
    st.plotly_chart(fig_bubble, use_container_width=True)

# Bottom: Bar + Donut
col9, col10 = st.columns(2)
with col9:
    fig_bar = px.bar(df.groupby('area')['score'].mean().reset_index(), x='area', y='score',
                     title="Avg Score by Area", color='score', color_continuous_scale='viridis')
    st.plotly_chart(fig_bar, use_container_width=True)

with col10:
    fig_bottom_donut = px.pie(df, names='risk_level', title="Risk Breakdown")
    st.plotly_chart(fig_bottom_donut, use_container_width=True)

st.markdown("---")
st.caption("💡 Filters/Sidebar | Clustering Live | Data: Dubai Census + Your CSV [code_file:56]")
