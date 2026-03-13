import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", page_title="Lenskart Intelligence", page_icon="🕶️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}
.block-container { padding: 1.5rem 2rem !important; }
h1, h2, h3 { color: white !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.12) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.2) !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #1e293b !important; font-weight: 700 !important; }
.stPlotlyChart {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
    padding: 0.5rem !important;
}
.stDataFrame {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: white !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.95) !important;
    color: #667eea !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── DATA ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("lenskart_dubai_stores.csv")
    df = df.fillna(0)
    return df

df = load_data()

FEATURES = ["pop_density", "med_age", "med_income_aed",
            "competitors", "footfall_daily", "near_mall", "metro_access"]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🕶️ **Lenskart AI**")
    st.markdown("*Dubai Expansion Intelligence*")
    st.markdown("---")

    st.markdown("### ⚙️ ML Controls")
    k = st.slider("KMeans Clusters", 3, 10, 5)
    threshold = st.slider("Optimal Score Threshold", 0.50, 0.90, 0.75)

    st.markdown("### 🔎 Filters")
    areas = sorted(df["area"].unique())
    selected_areas = st.multiselect("Areas", areas, default=areas[:4])
    df_f = df[df["area"].isin(selected_areas)] if selected_areas else df.copy()

    st.markdown("---")
    st.markdown("### 🔮 Site Predictor")
    pred_density   = st.number_input("Pop Density",      1.0, 50.0, 15.0)
    pred_income    = st.number_input("Income (AED)",  7500, 45000, 18000)
    pred_age       = st.number_input("Median Age",       20,    55,   32)
    pred_comp      = st.number_input("Competitors",       0,    10,    2)
    pred_footfall  = st.number_input("Daily Footfall",  200, 45000, 8000)
    pred_mall      = st.selectbox("Near Mall?", [1, 0])
    pred_metro     = st.selectbox("Metro Access?", [1, 0])
    predict_btn    = st.button("🚀 Predict Site", use_container_width=True)


# ── ML PIPELINE ───────────────────────────────────────────────────────────────

X = df_f[FEATURES].fillna(df_f[FEATURES].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_f = df_f.copy()
df_f["cluster"] = kmeans.fit_predict(X_scaled)

y = (df_f["composite_score"] > df_f["composite_score"].quantile(threshold)).astype(int)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
rf.fit(X_scaled, y)
df_f["success_prob"] = rf.predict_proba(X_scaled)[:, 1]

top5 = df_f.nlargest(5, "success_prob")[["area", "lat", "lon", "composite_score", "success_prob", "risk_level"]]


# ── HEADER ────────────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🕶️ Lenskart Dubai Intelligence System")
    st.markdown("*AI-powered outlet location optimizer — real-time predictions*")
with col_h2:
    st.metric("Sites Analyzed", f"{len(df_f):,}")


# ── KPI ROW ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("✅ Optimal Sites",      df_f["optimal_site"].sum())
c2.metric("🧬 Clusters",           k)
c3.metric("💰 Avg Score",          f"{df_f['composite_score'].mean():.2f}")
c4.metric("⚠️ High Risk",          (df_f["risk_level"] == "High").sum())
c5.metric("📈 Avg Success Prob",   f"{df_f['success_prob'].mean():.1%}")


# ── TABS ──────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs([
    "🗺️ Map View",
    "📊 Cluster Analysis",
    "🔗 Association Rules",
    "📉 Aging Matrix",
    "🎯 Predictor"
])


# ════════════════════════════════════════════════════
# TAB 1 — MAP VIEW
# ════════════════════════════════════════════════════
with t1:
    st.markdown("### 🗺️ Dubai Site Intelligence Map")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        fig_map = px.scatter_mapbox(
            df_f, lat="lat", lon="lon",
            color="cluster", size="composite_score",
            hover_name="area",
            hover_data={"risk_level": True, "success_prob": ":.1%",
                        "composite_score": ":.2f", "lat": False, "lon": False},
            color_continuous_scale="viridis",
            mapbox_style="carto-positron",
            zoom=10, height=520,
            title="Dubai Outlet Location Clusters"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    with col_m2:
        st.markdown("#### 🏆 Top 5 Sites")
        for i, row in top5.iterrows():
            st.markdown(f"""
            **{row['area']}**  
            Score: `{row['composite_score']:.2f}` | Prob: `{row['success_prob']:.1%}`  
            Risk: `{row['risk_level']}`  
            ---
            """)

    # Area bar chart
    area_avg = df_f.groupby("area")["composite_score"].mean().reset_index().sort_values("composite_score", ascending=False)
    fig_area = px.bar(area_avg, x="area", y="composite_score",
                      color="composite_score", color_continuous_scale="viridis",
                      title="Average Composite Score by Area", height=350)
    st.plotly_chart(fig_area, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 2 — CLUSTER ANALYSIS
# ════════════════════════════════════════════════════
with t2:
    st.markdown("### 🧬 Cluster Intelligence Profiles")

    col_cl1, col_cl2 = st.columns(2)

    with col_cl1:
        # Scatter
        fig_sc = px.scatter(
            df_f, x="med_income_aed", y="footfall_daily",
            color="cluster", size="composite_score",
            hover_name="area", title="Income vs Footfall by Cluster",
            height=400
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_cl2:
        # Radar for each cluster
        cluster_means = df_f.groupby("cluster")[FEATURES].mean()
        fig_radar = go.Figure()
        for cluster_id, row in cluster_means.iterrows():
            vals = row.tolist()
            vals_norm = [(v - min(vals)) / (max(vals) - min(vals) + 1e-9) for v in vals]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_norm + [vals_norm[0]],
                theta=FEATURES + [FEATURES[0]],
                fill="toself",
                name=f"Cluster {cluster_id}"
            ))
        fig_radar.update_layout(title="Cluster Radar Profiles", height=400,
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

    # Cluster summary table
    st.markdown("#### 📋 Cluster Summary")
    cluster_summary = df_f.groupby("cluster").agg(
        Sites=("composite_score", "count"),
        Avg_Score=("composite_score", "mean"),
        Avg_Income=("med_income_aed", "mean"),
        Avg_Footfall=("footfall_daily", "mean"),
        Optimal_Pct=("optimal_site", "mean"),
        Avg_Prob=("success_prob", "mean")
    ).round(2)
    cluster_summary["Optimal_Pct"] = (cluster_summary["Optimal_Pct"] * 100).round(1).astype(str) + "%"
    cluster_summary["Avg_Prob"] = (cluster_summary["Avg_Prob"] * 100).round(1).astype(str) + "%"
    st.dataframe(cluster_summary, use_container_width=True)

    # Feature importance
    st.markdown("#### 🎛️ Feature Importance (Random Forest)")
    imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}).sort_values("Importance")
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="plasma",
                     title="What Drives Optimal Sites?", height=350)
    st.plotly_chart(fig_imp, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 3 — ASSOCIATION RULES
# ════════════════════════════════════════════════════
with t3:
    st.markdown("### 🔗 Association Rules (Market Basket)")
    st.caption("Discovers patterns: e.g., {near_mall + high_income} → {optimal_site}")

    df_bin = df_f[["near_mall", "metro_access", "optimal_site"]].astype(bool)
    freq_sets = apriori(df_bin, min_support=0.1, use_colnames=True)

    if not freq_sets.empty:
        rules = association_rules(freq_sets, metric="lift", min_threshold=1.0).round(3)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Rules Found", len(rules))
        col_r2.metric("Max Lift", f"{rules['lift'].max():.2f}")
        col_r3.metric("Max Confidence", f"{rules['confidence'].max():.1%}")

        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]],
                     use_container_width=True)

        fig_rules = px.scatter(rules, x="support", y="confidence", size="lift",
                               color="lift", hover_data=["antecedents", "consequents"],
                               title="Rules: Support vs Confidence (size = Lift)",
                               color_continuous_scale="plasma", height=400)
        st.plotly_chart(fig_rules, use_container_width=True)
    else:
        st.info("No rules found. Select more areas in the sidebar.")


# ════════════════════════════════════════════════════
# TAB 4 — AGING MATRIX
# ════════════════════════════════════════════════════
with t4:
    st.markdown("### 📉 Site Aging Risk Matrix")
    st.caption("Shows how site risk evolves over a 3-year lease lifecycle")

    years = [1, 2, 3]
    risk_multipliers = {"Low": [1.0, 1.1, 1.2], "Medium": [1.0, 1.3, 1.6], "High": [1.0, 1.6, 2.2]}

    risk_counts = df_f["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    aging_rows = []
    for _, row in risk_counts.iterrows():
        rl = str(row["risk_level"])
        if rl in risk_multipliers:
            for yr, mult in zip(years, risk_multipliers[rl]):
                aging_rows.append({
                    "Risk Level": rl,
                    "Year": f"Year {yr}",
                    "Sites": int(row["count"]),
                    "Adjusted Risk Score": round(row["count"] * mult, 1)
                })

    aging_df = pd.DataFrame(aging_rows)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        fig_aging = px.bar(aging_df, x="Year", y="Adjusted Risk Score",
                           color="Risk Level", barmode="group",
                           color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
                           title="Risk Evolution Over 3-Year Lease", height=400)
        st.plotly_chart(fig_aging, use_container_width=True)

    with col_a2:
        fig_risk_pie = px.pie(df_f, names="risk_level", hole=0.45,
                              color="risk_level",
                              color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b",
                                                  "High": "#ef4444", "Critical": "#7c3aed"},
                              title="Current Risk Distribution", height=400)
        st.plotly_chart(fig_risk_pie, use_container_width=True)

    # Heatmap: Area x Risk
    st.markdown("#### 🔥 Area × Risk Level Heatmap")
    heat_df = df_f.groupby(["area", "risk_level"]).size().reset_index(name="count")
    heat_pivot = heat_df.pivot(index="area", columns="risk_level", values="count").fillna(0)
    fig_heat = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r",
                         title="Sites per Area by Risk Level", height=400, aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 5 — PREDICTOR
# ════════════════════════════════════════════════════
with t5:
    st.markdown("### 🎯 Live Site Predictor")
    st.caption("Enter site details in the sidebar → Click Predict")

    if predict_btn:
        new_site = np.array([[pred_density, pred_age, pred_income,
                              pred_comp, pred_footfall, pred_mall, pred_metro]])
        new_scaled = scaler.transform(new_site)
        new_cluster = kmeans.predict(new_scaled)[0]
        new_prob = rf.predict_proba(new_scaled)[0][1]
        new_score = (
            0.30 * np.log1p(pred_footfall) / 10 +
            0.25 * (1 if pred_age < 38 and pred_income > 15000 else 0) +
            0.20 * (1 / (1 + pred_comp)) +
            0.15 * (pred_mall + pred_metro) +
            0.10 * (pred_density / 20)
        )

        st.balloons()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Success Probability", f"{new_prob:.1%}")
        c2.metric("🧬 Assigned Cluster",    new_cluster)
        c3.metric("📊 Composite Score",     f"{new_score:.2f}")
        c4.metric("📌 Recommendation",      "✅ OPEN" if new_prob > 0.65 else "⚠️ REVIEW")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_prob * 100,
            title={"text": "Optimal Probability (%)"},
            delta={"reference": 65},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#667eea"},
                "steps": [
                    {"range": [0, 40],  "color": "#fee2e2"},
                    {"range": [40, 65], "color": "#fef3c7"},
                    {"range": [65, 100],"color": "#dcfce7"}
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 65}
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        if new_prob > 0.65:
            st.success(f"✅ **RECOMMENDED** — Cluster {new_cluster} | Score {new_score:.2f} | {new_prob:.1%} success probability")
        else:
            st.warning(f"⚠️ **REVIEW NEEDED** — Low probability {new_prob:.1%}. Consider adjusting location parameters.")
    else:
        st.info("👈 Set parameters in the **sidebar** → Click **Predict Site**")

    # Download
    st.markdown("---")
    st.markdown("### 💾 Export Insights")
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("📊 Download Full Analysis CSV", csv,
                       "lenskart_analysis.csv", "text/csv",
                       use_container_width=True)
