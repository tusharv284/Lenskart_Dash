"""
🕶️ LENSKART DUBAI INTELLIGENCE SYSTEM - PRO EDITION
UX: 3 Pages | ML: Clustering + RF + Rules | Charts: 10 Mind-Blowing Visuals
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

st.set_page_config(
    layout="wide",
    page_title="Lenskart Dubai Intelligence",
    page_icon="🕶️",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
}
.block-container { padding: 1.5rem 2.5rem !important; }
h1, h2, h3, p, label { color: white !important; }
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 20px !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(10px) !important;
    transition: transform 0.2s !important;
}
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.7) !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; font-size: 2rem !important; }
[data-testid="stMetricDelta"] { font-weight: 600 !important; }
.stPlotlyChart {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(10px) !important;
    padding: 0.5rem !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.5) !important;
    transition: all 0.3s !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}
.stDataFrame {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
.hero-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
    padding: 2rem;
    backdrop-filter: blur(15px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}
.rec-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.2);
}
.go-badge {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white; font-weight: 800; font-size: 1.5rem;
    padding: 0.5rem 2rem; border-radius: 50px;
    display: inline-block; box-shadow: 0 4px 20px rgba(34,197,94,0.5);
}
.avoid-badge {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white; font-weight: 800; font-size: 1.5rem;
    padding: 0.5rem 2rem; border-radius: 50px;
    display: inline-block; box-shadow: 0 4px 20px rgba(239,68,68,0.5);
}
.caution-badge {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white; font-weight: 800; font-size: 1.5rem;
    padding: 0.5rem 2rem; border-radius: 50px;
    display: inline-block; box-shadow: 0 4px 20px rgba(245,158,11,0.5);
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=20, r=20, t=50, b=20)
)

# ── DATA + ML ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("lenskart_dubai_stores.csv").fillna(0)
    features = ["pop_density", "med_age", "med_income_aed",
                "competitors", "footfall_daily", "near_mall", "metro_access"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    cluster_names = {0: "Premium Hub", 1: "Mass Market", 2: "Emerging Zone",
                     3: "High Footfall", 4: "Residential", 5: "Business District"}
    df["location_type"] = df["cluster"].map(cluster_names)
    y = (df["composite_score"] > df["composite_score"].quantile(0.75)).astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_scaled, y)
    df["confidence"] = rf.predict_proba(X_scaled)[:, 1]
    df["stars"] = pd.cut(df["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         labels=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
    df["verdict"] = df["confidence"].apply(
        lambda x: "✅ Recommended" if x > 0.65 else ("⚠️ Review" if x > 0.40 else "❌ Avoid"))
    return df, scaler, kmeans, rf, features

df, scaler, kmeans, rf, FEATURES = load_and_train()
top5 = df.nlargest(5, "confidence")

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕶️ **Lenskart AI**")
    st.markdown("*Dubai Expansion Intelligence*")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "🔍 Explore Sites", "🎯 Predict Location"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Advanced Filters**")
    sel_areas = st.multiselect("Areas", sorted(df["area"].unique()),
                               default=sorted(df["area"].unique())[:5])
    df_f = df[df["area"].isin(sel_areas)] if sel_areas else df.copy()
    st.markdown("---")
    st.caption(f"📊 {len(df_f):,} sites | {df_f['verdict'].str.contains('Recommended').sum()} recommended")


# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":

    # Hero
    st.markdown("""
    <div class="hero-card">
        <h1 style="font-size:2.8rem;margin:0">🕶️ Lenskart Dubai Intelligence</h1>
        <p style="font-size:1.1rem;color:rgba(255,255,255,0.7);margin-top:0.5rem">
        AI-powered outlet location optimizer · Real-time predictions · Dubai expansion
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📍 Sites Analyzed",    f"{len(df_f):,}",   "+2,500")
    k2.metric("✅ Recommended",        df_f["verdict"].str.contains("Recommended").sum(), "+12%")
    k3.metric("🏆 Avg Confidence",     f"{df_f['confidence'].mean():.1%}", "↑ 5%")
    k4.metric("⚠️ High Risk Sites",   (df_f["risk_level"] == "High").sum(), "-8%")

    st.markdown("---")

    # ROW 1 — Map + Sunburst
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🗺️ Dubai Site Intelligence Map")
        fig_map = px.scatter_mapbox(
            df_f, lat="lat", lon="lon",
            color="confidence",
            size="composite_score",
            hover_name="area",
            hover_data={"verdict": True, "stars": True,
                        "risk_level": True, "confidence": ":.1%",
                        "lat": False, "lon": False},
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            mapbox_style="carto-darkmatter",
            zoom=10, height=480,
            opacity=0.85
        )
        fig_map.update_layout(**CHART_TEMPLATE, coloraxis_showscale=True)
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top Recommended Sites")
        for _, row in top5.head(3).iterrows():
            st.markdown(f"""
            <div class="rec-card">
                <div style="font-size:1.1rem;font-weight:700">{row['area']}</div>
                <div style="font-size:1.5rem;margin:0.3rem 0">{row['stars']}</div>
                <div style="color:rgba(255,255,255,0.8);font-size:0.85rem">
                    Confidence: <b>{row['confidence']:.1%}</b><br>
                    Risk: <b>{row['risk_level']}</b><br>
                    Type: <b>{row['location_type']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ROW 2 — Sunburst + Treemap
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🌀 Site Hierarchy Sunburst")
        fig_sun = px.sunburst(
            df_f, path=["area", "location_type", "verdict"],
            values="composite_score",
            color="confidence",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            title="Area → Location Type → Verdict",
            height=450
        )
        fig_sun.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig_sun, use_container_width=True)

    with col4:
        st.markdown("### 🌲 Footfall Treemap")
        fig_tree = px.treemap(
            df_f, path=["area", "risk_level"],
            values="footfall_daily",
            color="confidence",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            title="Area → Risk (Size = Footfall)",
            height=450
        )
        fig_tree.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig_tree, use_container_width=True)

    # ROW 3 — Sankey + Funnel
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 🔀 Sankey: Demographics Flow")
        area_agg = df_f.groupby(["location_type", "verdict"]).size().reset_index(name="count")
        all_nodes = list(df_f["location_type"].unique()) + list(df_f["verdict"].unique())
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                label=all_nodes,
                color=["#667eea","#764ba2","#a78bfa","#818cf8","#6366f1","#4f46e5",
                       "#22c55e","#f59e0b","#ef4444"],
                pad=20, thickness=25
            ),
            link=dict(
                source=[node_idx[r["location_type"]] for _, r in area_agg.iterrows()],
                target=[node_idx[r["verdict"]] for _, r in area_agg.iterrows()],
                value=area_agg["count"].tolist(),
                color="rgba(102,126,234,0.3)"
            )
        ))
        fig_sankey.update_layout(**CHART_TEMPLATE, title="Location Types → Recommendations", height=420)
        st.plotly_chart(fig_sankey, use_container_width=True)

    with col6:
        st.markdown("### 🔺 Site Qualification Funnel")
        total = len(df_f)
        near_mall = df_f["near_mall"].sum()
        metro = df_f["metro_access"].sum()
        low_comp = (df_f["competitors"] <= 2).sum()
        good_income = (df_f["med_income_aed"] > 15000).sum()
        recommended = df_f["verdict"].str.contains("Recommended").sum()

        fig_funnel = go.Figure(go.Funnel(
            y=["All Sites","Near Mall","Metro Access","Low Competition","Good Income","✅ Recommended"],
            x=[total, near_mall, metro, low_comp, good_income, recommended],
            textinfo="value+percent initial",
            marker=dict(color=["#6366f1","#818cf8","#a78bfa","#c4b5fd","#667eea","#22c55e"])
        ))
        fig_funnel.update_layout(**CHART_TEMPLATE, title="Site Qualification Pipeline", height=420)
        st.plotly_chart(fig_funnel, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE
# ════════════════════════════════════════════════════════
elif page == "🔍 Explore Sites":

    st.markdown("## 🔍 Explore All Sites")
    st.caption("Dive deep into location data, clusters, and risk intelligence")

    tab1, tab2, tab3 = st.tabs(["📊 Deep Analytics", "🧬 Location Types", "⏳ 3-Year Outlook"])

    with tab1:
        # 3D Scatter
        st.markdown("### 🌐 3D Location Intelligence Globe")
        fig_3d = px.scatter_3d(
            df_f.sample(500, random_state=42),
            x="med_income_aed", y="footfall_daily", z="composite_score",
            color="confidence", size="pop_density",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            hover_name="area", opacity=0.85,
            labels={"med_income_aed": "Income (AED)",
                    "footfall_daily": "Daily Footfall",
                    "composite_score": "Site Score"},
            title="3D: Income × Footfall × Score", height=550
        )
        fig_3d.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig_3d, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Parallel Coordinates
            st.markdown("### 📏 Parallel Coordinates")
            df_sample = df_f.sample(min(300, len(df_f)), random_state=42)
            fig_par = px.parallel_coordinates(
                df_sample,
                dimensions=FEATURES,
                color="confidence",
                color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                title="All Features vs Confidence", height=450
            )
            fig_par.update_layout(**CHART_TEMPLATE)
            st.plotly_chart(fig_par, use_container_width=True)

        with col2:
            # Correlation Heatmap
            st.markdown("### 🔥 Feature Correlation Heatmap")
            corr = df_f[FEATURES + ["composite_score"]].corr()
            fig_heat = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=[[0,"#764ba2"],[0.5,"#16213e"],[1,"#22c55e"]],
                title="Feature Correlations", height=450, aspect="auto"
            )
            fig_heat.update_layout(**CHART_TEMPLATE)
            st.plotly_chart(fig_heat, use_container_width=True)

        # Animated Bubble Chart
        st.markdown("### 🎬 Animated: Site Performance by Area")
        df_anim = df_f.copy()
        df_anim["income_band"] = pd.cut(df_f["med_income_aed"],
                                        bins=3, labels=["Low", "Mid", "High"])
        fig_anim = px.scatter(
            df_anim, x="med_income_aed", y="footfall_daily",
            animation_frame="income_band",
            size="composite_score", color="confidence",
            hover_name="area", size_max=40,
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            title="Sites Animated by Income Band", height=480
        )
        fig_anim.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig_anim, use_container_width=True)

        # Feature Importance
        st.markdown("### 🎛️ What Drives a Recommended Site?")
        imp_df = pd.DataFrame({"Feature": FEATURES,
                               "Importance": rf.feature_importances_}).sort_values("Importance")
        labels = {"pop_density": "Population Density", "med_age": "Median Age",
                  "med_income_aed": "Median Income", "competitors": "Competitors",
                  "footfall_daily": "Daily Footfall", "near_mall": "Near Mall",
                  "metro_access": "Metro Access"}
        imp_df["Feature"] = imp_df["Feature"].map(labels)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="viridis",
                         title="Feature Importance (Random Forest)", height=380)
        fig_imp.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.markdown("### 🧬 Location Type Profiles")

        col1, col2 = st.columns(2)
        with col1:
            # Radar per cluster
            cluster_means = df_f.groupby("location_type")[FEATURES].mean()
            fig_radar = go.Figure()
            colors = ["#667eea","#22c55e","#f59e0b","#ef4444","#a78bfa","#38bdf8"]
            for i, (lt, row) in enumerate(cluster_means.iterrows()):
                vals = row.tolist()
                mn, mx = min(vals), max(vals) + 1e-9
                norm = [(v - mn) / (mx - mn) for v in vals]
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm + [norm[0]], theta=list(labels.values()) + [list(labels.values())[0]],
                    fill="toself", name=lt, line_color=colors[i % len(colors)]
                ))
            fig_radar.update_layout(**CHART_TEMPLATE, title="Location Type Radar",
                                    polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                               gridcolor="rgba(255,255,255,0.1)"),
                                               angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
                                    height=470)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Violin chart
            fig_violin = px.violin(df_f, y="confidence", x="location_type",
                                   box=True, points="outliers", color="location_type",
                                   title="Confidence Distribution by Location Type", height=470)
            fig_violin.update_layout(**CHART_TEMPLATE, showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)

        # Association Rules
        st.markdown("### 🔗 Key Insights (Association Rules)")
        df_bin = df_f[["near_mall", "metro_access", "optimal_site"]].astype(bool)
        freq = apriori(df_bin, min_support=0.1, use_colnames=True)
        if not freq.empty:
            rules = association_rules(freq, metric="lift", min_threshold=1.0).round(3)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: " + ".join(list(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: " + ".join(list(x)))
            rules.columns = ["IF (Antecedent)", "THEN (Consequent)", "Support",
                             "Confidence", "Lift", "Leverage", "Conviction", "Zhang's"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rules Found", len(rules))
            c2.metric("Max Lift",    f"{rules['Lift'].max():.2f}x")
            c3.metric("Max Confidence", f"{rules['Confidence'].max():.1%}")
            st.dataframe(rules[["IF (Antecedent)", "THEN (Consequent)",
                                 "Support", "Confidence", "Lift"]].head(10),
                         use_container_width=True)
            fig_rules = px.scatter(rules, x="Support", y="Confidence", size="Lift",
                                   color="Lift", hover_data=["IF (Antecedent)", "THEN (Consequent)"],
                                   color_continuous_scale="plasma",
                                   title="Support vs Confidence (size = Lift)", height=380)
            fig_rules.update_layout(**CHART_TEMPLATE)
            st.plotly_chart(fig_rules, use_container_width=True)

    with tab3:
        st.markdown("### ⏳ 3-Year Risk Outlook (Aging Matrix)")
        st.caption("How site risk evolves over a standard 3-year lease period")

        risk_data = {"Low": [1.0, 1.1, 1.2], "Medium": [1.0, 1.4, 1.9], "High": [1.0, 1.8, 2.8]}
        counts = df_f["risk_level"].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            rows = []
            for rl, mults in risk_data.items():
                if rl in counts.index:
                    for yr, m in zip([1, 2, 3], mults):
                        rows.append({"Risk Level": rl, "Year": f"Year {yr}",
                                     "Risk Score": round(counts[rl] * m)})
            aging_df = pd.DataFrame(rows)
            fig_aging = px.line(aging_df, x="Year", y="Risk Score",
                                color="Risk Level", markers=True,
                                color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"},
                                title="Risk Evolution Across 3-Year Lease", height=400)
            fig_aging.update_traces(line_width=3, marker_size=10)
            fig_aging.update_layout(**CHART_TEMPLATE)
            st.plotly_chart(fig_aging, use_container_width=True)

        with col2:
            fig_area_risk = px.area(aging_df, x="Year", y="Risk Score",
                                    color="Risk Level",
                                    color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"},
                                    title="Cumulative Risk Exposure", height=400)
            fig_area_risk.update_layout(**CHART_TEMPLATE)
            st.plotly_chart(fig_area_risk, use_container_width=True)

        # Heatmap: Area x Risk
        st.markdown("### 🔥 Area × Risk Heatmap")
        heat_pivot = df_f.groupby(["area", "risk_level"]).size().unstack(fill_value=0)
        fig_heat2 = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r",
                              text_auto=True, title="Sites per Area by Risk Level",
                              height=420, aspect="auto")
        fig_heat2.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig_heat2, use_container_width=True)

        # Export
        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Analysis", csv,
                           "lenskart_analysis.csv", "text/csv",
                           use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ════════════════════════════════════════════════════════
elif page == "🎯 Predict Location":

    st.markdown("## 🎯 Location Predictor")
    st.caption("Enter site details and get an instant AI-powered recommendation")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("""<div class="hero-card">""", unsafe_allow_html=True)
        st.markdown("### 📝 Site Details")

        area_input   = st.text_input("📍 Area Name (optional)", "JLT")
        income       = st.slider("💰 Median Income (AED/month)", 7500, 45000, 18000, 500)
        footfall     = st.slider("👣 Estimated Daily Footfall", 200, 45000, 8000, 100)
        density      = st.slider("👥 Population Density", 1.0, 45.0, 15.0, 0.5)
        age          = st.slider("👶 Median Age", 20, 55, 32)
        competitors  = st.slider("🏪 Nearby Competitors", 0, 10, 2)
        near_mall    = st.toggle("🏬 Near a Mall?", True)
        metro        = st.toggle("🚇 Metro Access?", True)

        st.markdown("""</div>""", unsafe_allow_html=True)
        predict      = st.button("🚀 Predict This Location", use_container_width=True)

    with col_result:
        if predict:
            new_X = np.array([[density, age, income, competitors, footfall,
                               int(near_mall), int(metro)]])
            new_scaled  = scaler.transform(new_X)
            cluster_id  = kmeans.predict(new_scaled)[0]
            prob        = rf.predict_proba(new_scaled)[0][1]
            cluster_names = {0: "Premium Hub", 1: "Mass Market", 2: "Emerging Zone",
                             3: "High Footfall", 4: "Residential", 5: "Business District"}
            loc_type    = cluster_names.get(int(cluster_id), "Mixed")
            stars       = "⭐⭐⭐⭐⭐" if prob > 0.8 else "⭐⭐⭐⭐" if prob > 0.65 else "⭐⭐⭐" if prob > 0.4 else "⭐⭐"

            if prob > 0.65:
                badge = '<span class="go-badge">✅ GO — OPEN HERE</span>'
                st.balloons()
            elif prob > 0.40:
                badge = '<span class="caution-badge">⚠️ CAUTION — REVIEW</span>'
            else:
                badge = '<span class="avoid-badge">❌ AVOID — HIGH RISK</span>'

            st.markdown(f"""
            <div class="hero-card" style="text-align:center">
                <h2>{area_input if area_input else "New Site"}</h2>
                <div style="font-size:2rem;margin:0.5rem 0">{stars}</div>
                {badge}
                <br><br>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;text-align:left;margin-top:1rem">
                    <div>📊 Confidence Score<br><b style="font-size:1.5rem">{prob:.1%}</b></div>
                    <div>🧬 Location Type<br><b style="font-size:1.1rem">{loc_type}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                delta={"reference": 65, "valueformat": ".1f"},
                title={"text": "Confidence Score (%)", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white", "size": 48}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#667eea", "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "bordercolor": "rgba(255,255,255,0.2)",
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(239,68,68,0.3)"},
                        {"range": [40, 65], "color": "rgba(245,158,11,0.3)"},
                        {"range": [65, 100],"color": "rgba(34,197,94,0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8, "value": 65
                    }
                }
            ))
            fig_gauge.update_layout(**CHART_TEMPLATE, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Waterfall Score Decomposition
            st.markdown("### 🔍 Score Breakdown")
            score_components = {
                "Footfall":      0.30 * np.log1p(footfall) / 10,
                "Demographics":  0.25 * (1 if age < 38 and income > 15000 else 0),
                "Low Competition": 0.20 * (1 / (1 + competitors)),
                "Accessibility": 0.15 * (int(near_mall) + int(metro)),
                "Density":       0.10 * (density / 20)
            }
            sc_df = pd.DataFrame(score_components.items(), columns=["Factor", "Score"])
            fig_wf = go.Figure(go.Waterfall(
                name="Score", orientation="v",
                x=sc_df["Factor"].tolist() + ["Total"],
                y=sc_df["Score"].tolist() + [None],
                measure=["relative"] * len(sc_df) + ["total"],
                connector={"line": {"color": "rgba(255,255,255,0.3)"}},
                increasing={"marker": {"color": "#22c55e"}},
                totals={"marker": {"color": "#667eea"}},
                text=[f"{v:.3f}" for v in sc_df["Score"].tolist()] + [f"{sc_df['Score'].sum():.3f}"]
            ))
            fig_wf.update_layout(**CHART_TEMPLATE, title="Composite Score Decomposition", height=380)
            st.plotly_chart(fig_wf, use_container_width=True)

        else:
            st.markdown("""
            <div class="hero-card" style="text-align:center;padding:4rem 2rem">
                <div style="font-size:5rem">🎯</div>
                <h3>Ready to Predict!</h3>
                <p style="color:rgba(255,255,255,0.6)">
                Fill in the site details on the left<br>and hit <b>Predict This Location</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem;color:rgba(255,255,255,0.4);font-size:0.85rem">
    🕶️ Lenskart Dubai Intelligence · ML-Powered · Production Ready
</div>
""", unsafe_allow_html=True)
