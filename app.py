"""
🕶️ LENSKART DUBAI INTELLIGENCE SYSTEM - WITH AI CHATBOT
3 Pages + Chatbot | 10 Charts | ML: Clustering + RF + Association Rules
Every chart has a plain-English inference box below it.
Data: lenskart_dubai_stores.csv
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
import re

st.set_page_config(
    layout="wide",
    page_title="Lenskart Dubai Intelligence",
    page_icon="🕶️",
    initial_sidebar_state="collapsed"
)

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
}
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.7) !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; font-size: 2rem !important; }
.stPlotlyChart {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    padding: 0.5rem !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.7rem 2rem !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.5) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 14px !important; padding: 5px !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 600 !important; border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}
.hero-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px; padding: 2rem;
    backdrop-filter: blur(15px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}
.rec-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px; padding: 1.5rem; text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.2); margin-bottom: 1rem;
}
.inference-box {
    background: rgba(102,126,234,0.12);
    border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 1.5rem 0;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.92rem; line-height: 1.6;
}
.inference-box b { color: #a78bfa !important; }
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
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    margin-bottom: 0.5rem !important;
    padding: 0.5rem !important;
}
[data-testid="stChatInputTextArea"] {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 14px !important;
}
.suggested-btn > button {
    background: rgba(102,126,234,0.2) !important;
    border: 1px solid rgba(102,126,234,0.4) !important;
    color: white !important; border-radius: 20px !important;
    font-size: 0.8rem !important; padding: 0.3rem 0.8rem !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


def inference(text):
    st.markdown(f'''<div class="inference-box">💡 <b>What this tells us:</b> {text}</div>''',
                unsafe_allow_html=True)


CHART_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"), margin=dict(l=20, r=20, t=50, b=20)
)
AXIS = dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)")

FEATURES = ["pop_density", "med_age", "med_income_aed",
            "competitors", "footfall_daily", "near_mall", "metro_access"]
FEATURE_LABELS = {
    "pop_density": "Population Density", "med_age": "Median Age",
    "med_income_aed": "Median Income",   "competitors": "Competitors",
    "footfall_daily": "Daily Footfall",  "near_mall": "Near Mall",
    "metro_access": "Metro Access"
}
CLUSTER_NAMES = {
    0: "Premium Hub", 1: "Mass Market", 2: "Emerging Zone",
    3: "High Footfall", 4: "Residential", 5: "Business District"
}
COLORS = ["#667eea","#22c55e","#f59e0b","#ef4444","#a78bfa","#38bdf8"]


@st.cache_data
def load_and_train():
    df = pd.read_csv("lenskart_dubai_stores.csv").fillna(0)
    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    df["location_type"] = df["cluster"].map(CLUSTER_NAMES)
    y = (df["composite_score"] > df["composite_score"].quantile(0.75)).astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_scaled, y)
    df["confidence"] = rf.predict_proba(X_scaled)[:, 1]
    df["stars"] = pd.cut(df["confidence"], bins=[0,0.2,0.4,0.6,0.8,1.0],
                         labels=["⭐","⭐⭐","⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐⭐⭐"])
    df["verdict"] = df["confidence"].apply(
        lambda x: "✅ Recommended" if x > 0.65 else ("⚠️ Review" if x > 0.40 else "❌ Avoid")
    )
    return df, scaler, kmeans, rf


df, scaler, kmeans, rf = load_and_train()


# ══════════════════════════════════════════════════
# CHATBOT BRAIN — data-aware rule-based responses
# ══════════════════════════════════════════════════
def chatbot_response(user_msg: str, df: pd.DataFrame) -> str:
    msg = user_msg.lower().strip()

    top5        = df.nlargest(5, "confidence")
    total       = len(df)
    rec_count   = df["verdict"].str.contains("Recommended").sum()
    avoid_count = df["verdict"].str.contains("Avoid").sum()
    best_area   = df.groupby("area")["confidence"].mean().idxmax()
    worst_area  = df.groupby("area")["confidence"].mean().idxmin()
    best_score  = df["confidence"].max()
    avg_score   = df["confidence"].mean()
    high_risk   = (df["risk_level"] == "High").sum()
    low_risk    = (df["risk_level"] == "Low").sum()
    top_feature = FEATURE_LABELS[FEATURES[rf.feature_importances_.argmax()]]

    # ── Greetings ──
    if re.search(r"\b(hi|hello|hey|hiya|howdy|good morning|good evening)\b", msg):
        return (
            "👋 Hello! I'm **Lenskart AI**, your Dubai expansion assistant. "
            "I can help you find the best outlet locations, understand risks, "
            "interpret charts, and much more.\n\n"
            "Try asking me:\n"
            "- *What are the top recommended sites?*\n"
            "- *Which area has the highest risk?*\n"
            "- *What drives a successful outlet?*"
        )

    # ── Top sites ──
    if re.search(r"top|best|recommend|highest|ideal|where.*open|open.*where", msg):
        lines = "\n".join(
            [f"**{i+1}. {r['area']}** — {r['confidence']:.1%} confidence | {r['stars']} | {r['location_type']}"
             for i, (_, r) in enumerate(top5.head(5).iterrows())]
        )
        return (
            f"🏆 **Top 5 Recommended Sites in Dubai:**\n\n{lines}\n\n"
            f"➡️ **{best_area}** has the highest average confidence score across all its sites. "
            f"Focus expansion efforts there first!"
        )

    # ── Worst / avoid ──
    if re.search(r"worst|avoid|bad|risky|dangerous|lowest|not open", msg):
        bottom5 = df.nsmallest(5, "confidence")
        lines = "\n".join(
            [f"**{i+1}. {r['area']}** — {r['confidence']:.1%} confidence | ⚠️ {r['risk_level']} risk"
             for i, (_, r) in enumerate(bottom5.iterrows())]
        )
        return (
            f"⚠️ **Sites to Avoid:**\n\n{lines}\n\n"
            f"**{worst_area}** has the lowest average confidence. "
            f"High competitor density and low footfall are usually the main culprits."
        )

    # ── Risk ──
    if re.search(r"risk|danger|safe|safety|risky|low risk|high risk", msg):
        return (
            f"🔴 **Risk Breakdown across {total:,} sites:**\n\n"
            f"- 🟢 **Low Risk:** {low_risk} sites — safe to open, stable long-term\n"
            f"- 🟡 **Medium Risk:** {(df['risk_level'] == 'Medium').sum()} sites — viable with caution\n"
            f"- 🔴 **High Risk:** {high_risk} sites — avoid or do thorough feasibility study\n\n"
            f"💡 High-risk sites compound to **2.8× their current risk score by Year 3** of a lease. "
            f"Always check the *3-Year Outlook* tab before signing a long-term lease."
        )

    # ── Summary / overview ──
    if re.search(r"summar|overview|total|how many|count|dataset|data", msg):
        return (
            f"📊 **Dashboard Summary:**\n\n"
            f"- 📍 Total sites analysed: **{total:,}**\n"
            f"- ✅ Recommended: **{rec_count}** ({rec_count/total:.1%})\n"
            f"- ❌ Avoid: **{avoid_count}** ({avoid_count/total:.1%})\n"
            f"- 🏆 Best area: **{best_area}**\n"
            f"- 📈 Average confidence: **{avg_score:.1%}**\n"
            f"- ⚠️ High-risk sites: **{high_risk}**\n\n"
            f"Use the **🏠 Overview** page for a visual summary of all these metrics."
        )

    # ── Feature importance / drivers ──
    if re.search(r"driver|factor|import|what matter|what drive|key|influence|affect|weight", msg):
        imp_sorted = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
        lines = "\n".join(
            [f"**{i+1}. {FEATURE_LABELS[f]}** — {v:.1%} importance"
             for i, (f, v) in enumerate(imp_sorted)]
        )
        return (
            f"🎛️ **What Drives a Recommended Site?**\n\n{lines}\n\n"
            f"💡 **{top_feature}** is the single biggest predictor of outlet success. "
            f"When scouting any new Dubai location, prioritise this factor above all others."
        )

    # ── Confidence / score ──
    if re.search(r"confidence|score|probabilit|accuracy|model|predict|ml|machine learning|ai", msg):
        return (
            f"🤖 **About the AI Model:**\n\n"
            f"The dashboard uses a **Random Forest Classifier** trained on {total:,} Dubai sites.\n\n"
            f"- **Confidence Score** = probability that a site will be optimal (0–100%)\n"
            f"- **Threshold:** Sites above **65%** are ✅ Recommended\n"
            f"- **40–65%:** ⚠️ Review carefully\n"
            f"- **Below 40%:** ❌ Avoid\n\n"
            f"Average confidence across filtered sites: **{avg_score:.1%}**\n"
            f"Best site confidence: **{best_score:.1%}**"
        )

    # ── Location types / clusters ──
    if re.search(r"cluster|type|zone|category|segment|premium|mass market|emerging|residential|business", msg):
        counts = df["location_type"].value_counts()
        lines = "\n".join([f"- **{k}:** {v} sites" for k, v in counts.items()])
        return (
            f"🧬 **Location Types (KMeans Clusters):**\n\n{lines}\n\n"
            f"💡 **Premium Hub** and **High Footfall** zones consistently score the highest. "
            f"**Residential** zones tend to have lower confidence due to lower footfall "
            f"and less commercial traffic. Check the *🧬 Location Types* tab for radar charts."
        )

    # ── Footfall ──
    if re.search(r"footfall|traffic|people|crowd|busy|walk", msg):
        top_foot = df.nlargest(3, "footfall_daily")[["area","footfall_daily","confidence"]]
        lines = "\n".join(
            [f"- **{r['area']}:** {r['footfall_daily']:,.0f} daily visitors | {r['confidence']:.1%} confidence"
             for _, r in top_foot.iterrows()]
        )
        return (
            f"👣 **Highest Footfall Sites:**\n\n{lines}\n\n"
            f"💡 High footfall is one of the top drivers of outlet success. "
            f"However, combine it with low competition — high footfall + many rivals "
            f"still results in lower confidence scores."
        )

    # ── Competition ──
    if re.search(r"competi|rival|opponent|other store|nearby store", msg):
        low_comp = (df["competitors"] <= 1).sum()
        high_comp = (df["competitors"] >= 4).sum()
        return (
            f"🏪 **Competition Analysis:**\n\n"
            f"- **{low_comp}** sites have ≤1 nearby competitor (ideal)\n"
            f"- **{high_comp}** sites have 4+ competitors (saturated — avoid)\n\n"
            f"💡 Competitor count negatively correlates with confidence score. "
            f"Every additional competitor reduces site attractiveness. "
            f"Target areas where Lenskart can be the **dominant eyewear brand**."
        )

    # ── How to use ──
    if re.search(r"how.*use|how.*work|navigate|page|tab|help|guide|explain|tutorial", msg):
        return (
            "🗺️ **How to Use This Dashboard:**\n\n"
            "**🏠 Overview** — Big picture: map, top picks, sunburst, treemap, sankey, funnel\n"
            "**🔍 Explore Sites** — Deep dive: 3D chart, parallel coords, heatmap, animated bubble, clusters, association rules, 3-year risk\n"
            "**🎯 Predict Location** — Enter any site details → instant GO/CAUTION/AVOID verdict + gauge + waterfall breakdown\n"
            "**🤖 AI Assistant** — That's me! Ask anything about the data\n\n"
            "💡 Use the **sidebar filters** to narrow down to specific Dubai areas."
        )

    # ── Income / demographics ──
    if re.search(r"income|salary|earn|wealth|affluent|demographic|age|median", msg):
        avg_inc = df["med_income_aed"].mean()
        avg_age = df["med_age"].mean()
        return (
            f"💰 **Demographic Insights:**\n\n"
            f"- Average median income across sites: **AED {avg_inc:,.0f}/month**\n"
            f"- Average median age: **{avg_age:.1f} years**\n\n"
            f"💡 Lenskart's sweet spot is areas with **income > AED 15,000/month** and "
            f"**median age under 38** — young, affluent expats who are frequent eyewear buyers. "
            f"These demographics significantly boost the confidence score."
        )

    # ── Metro / mall ──
    if re.search(r"metro|mall|transport|access|location|proximity", msg):
        mall_pct  = df["near_mall"].mean()
        metro_pct = df["metro_access"].mean()
        both      = ((df["near_mall"] == 1) & (df["metro_access"] == 1)).sum()
        return (
            f"🚇 **Accessibility Analysis:**\n\n"
            f"- **{mall_pct:.1%}** of sites are near a mall\n"
            f"- **{metro_pct:.1%}** have metro access\n"
            f"- **{both}** sites have BOTH (prime locations)\n\n"
            f"💡 Sites with both mall proximity AND metro access show the **highest "
            f"association rule lift** — this combination is a strong predictor of success. "
            f"Always prioritise these dual-access sites."
        )

    # ── Association rules ──
    if re.search(r"rule|association|pattern|if.*then|basket|apriori", msg):
        return (
            "🔗 **Association Rule Insights:**\n\n"
            "The dashboard uses **Apriori algorithm** to find patterns like:\n\n"
            "- IF *near_mall* + *metro_access* → THEN *optimal_site* (High Lift)\n"
            "- IF *metro_access* → THEN *optimal_site* (Moderate Lift)\n\n"
            "💡 **Lift > 1.0** means the combination is more powerful than chance. "
            "These rules work like a quick checklist — sites ticking more rule conditions "
            "are significantly more likely to be recommended. "
            "See the *🧬 Location Types* tab for the full rules table."
        )

    # ── Fallback ──
    suggestions = [
        "top recommended sites", "risk analysis", "what drives success",
        "how to use the dashboard", "income demographics", "competition analysis"
    ]
    return (
        f"🤔 I'm not sure about that specific question, but I can help with:\n\n"
        + "\n".join([f"- *{s}*" for s in suggestions])
        + f"\n\nOr try rephrasing — I understand questions about sites, risks, "
        f"scores, locations, features, and how to use the dashboard!"
    )


# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕶️ **Lenskart AI**")
    st.markdown("*Dubai Expansion Intelligence*")
    st.markdown("---")
    page = st.radio("Navigate",
                    ["🏠 Overview", "🔍 Explore Sites", "🎯 Predict Location", "🤖 AI Assistant"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Filter Areas**")
    areas = sorted(df["area"].unique())
    sel   = st.multiselect("Areas", areas, default=areas[:5])
    df_f  = df[df["area"].isin(sel)].copy() if sel else df.copy()
    st.markdown("---")
    rec = df_f["verdict"].str.contains("Recommended").sum()
    st.caption(f"📊 {len(df_f):,} sites | ✅ {rec} recommended")


# ════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown("""
    <div class="hero-card">
        <h1 style="font-size:2.8rem;margin:0">🕶️ Lenskart Dubai Intelligence</h1>
        <p style="font-size:1.1rem;color:rgba(255,255,255,0.7);margin-top:0.5rem">
        AI-powered outlet location optimizer · Real-time ML predictions · Dubai expansion engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📍 Sites Analyzed",  f"{len(df_f):,}", "+2,500")
    k2.metric("✅ Recommended",      df_f["verdict"].str.contains("Recommended").sum(), "+12%")
    k3.metric("🏆 Avg Confidence",   f"{df_f['confidence'].mean():.1%}", "↑5%")
    k4.metric("⚠️ High Risk",        (df_f["risk_level"] == "High").sum(), "-8%")

    st.markdown("---")
    top5 = df_f.nlargest(5, "confidence")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🗺️ Dubai Site Intelligence Map")
        fig_map = px.scatter_mapbox(
            df_f, lat="lat", lon="lon", color="confidence", size="composite_score",
            hover_name="area",
            hover_data={"verdict": True, "stars": True, "risk_level": True,
                        "confidence": ":.1%", "lat": False, "lon": False},
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            mapbox_style="carto-darkmatter", zoom=10, height=480, opacity=0.85
        )
        fig_map.update_layout(**CHART_BG)
        st.plotly_chart(fig_map, use_container_width=True)
        inference("Each dot = a potential outlet location. <b>Green = high confidence (open here)</b>, "
                  "red = avoid. Larger dots have higher composite scores. "
                  "Focus on green clusters near business/mall zones for maximum success.")

    with col2:
        st.markdown("### 🏆 Top Recommendations")
        for _, row in top5.head(3).iterrows():
            st.markdown(f"""
            <div class="rec-card">
                <div style="font-size:1.1rem;font-weight:700">{row["area"]}</div>
                <div style="font-size:1.5rem;margin:0.3rem 0">{row["stars"]}</div>
                <div style="color:rgba(255,255,255,0.8);font-size:0.85rem">
                    Confidence: <b>{row["confidence"]:.1%}</b><br>
                    Risk: <b>{row["risk_level"]}</b><br>
                    Type: <b>{row["location_type"]}</b>
                </div>
            </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🌀 Sunburst Hierarchy")
        fig_sun = px.sunburst(
            df_f, path=["area","location_type","verdict"],
            values="composite_score", color="confidence",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            title="Area → Type → Verdict", height=450
        )
        fig_sun.update_layout(**CHART_BG)
        st.plotly_chart(fig_sun, use_container_width=True)
        inference("Drills from <b>Area → Location Type → Verdict</b>. Larger greener slices = "
                  "more high-scoring sites in that zone. Prioritise those areas first.")

    with col4:
        st.markdown("### 🌲 Footfall Treemap")
        fig_tree = px.treemap(
            df_f, path=["area","risk_level"], values="footfall_daily", color="confidence",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            title="Area → Risk (Size = Footfall)", height=450
        )
        fig_tree.update_layout(**CHART_BG)
        st.plotly_chart(fig_tree, use_container_width=True)
        inference("<b>Box size = daily footfall</b>. Colour = confidence. "
                  "Target: a large green box = high traffic + high success probability. "
                  "Avoid large red boxes — high traffic but too competitive.")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("### 🔀 Sankey: Location Flow")
        agg = df_f.groupby(["location_type","verdict"]).size().reset_index(name="count")
        all_nodes = list(df_f["location_type"].unique()) + list(df_f["verdict"].unique())
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        fig_sankey = go.Figure(go.Sankey(
            node=dict(label=all_nodes, color=COLORS[:len(all_nodes)], pad=20, thickness=25),
            link=dict(
                source=[node_idx[r["location_type"]] for _, r in agg.iterrows()],
                target=[node_idx[r["verdict"]] for _, r in agg.iterrows()],
                value=agg["count"].tolist(), color="rgba(102,126,234,0.3)"
            )
        ))
        fig_sankey.update_layout(**CHART_BG, title="Location Types → Recommendation", height=420)
        st.plotly_chart(fig_sankey, use_container_width=True)
        inference("<b>Thick flows into ✅ Recommended = that zone type consistently works.</b> "
                  "Thin flows into ❌ Avoid = skip that zone type in future site searches.")

    with col6:
        st.markdown("### 🔺 Site Qualification Funnel")
        total_f    = len(df_f)
        near_mall  = int(df_f["near_mall"].sum())
        metro_c    = int(df_f["metro_access"].sum())
        low_comp   = int((df_f["competitors"] <= 2).sum())
        good_inc   = int((df_f["med_income_aed"] > 15000).sum())
        rec_f      = int(df_f["verdict"].str.contains("Recommended").sum())
        fig_funnel = go.Figure(go.Funnel(
            y=["All Sites","Near Mall","Metro Access","Low Competition","Good Income","✅ Recommended"],
            x=[total_f, near_mall, metro_c, low_comp, good_inc, rec_f],
            textinfo="value+percent initial",
            marker=dict(color=["#6366f1","#818cf8","#a78bfa","#c4b5fd","#667eea","#22c55e"])
        ))
        fig_funnel.update_layout(**CHART_BG, title="Site Qualification Pipeline", height=420)
        st.plotly_chart(fig_funnel, use_container_width=True)
        inference("Each stage = one quality filter. <b>The biggest drop-off = the hardest "
                  "constraint to meet in Dubai.</b> Only sites passing all filters reach ✅ Recommended.")


# ════════════════════════════════════════════════
# PAGE 2 — EXPLORE
# ════════════════════════════════════════════════
elif page == "🔍 Explore Sites":

    st.markdown("## 🔍 Explore All Sites")
    tab1, tab2, tab3 = st.tabs(["📊 Deep Analytics", "🧬 Location Types", "⏳ 3-Year Outlook"])

    with tab1:
        st.markdown("### 🌐 3D Location Intelligence")
        fig_3d = px.scatter_3d(
            df_f.sample(min(500, len(df_f)), random_state=42),
            x="med_income_aed", y="footfall_daily", z="composite_score",
            color="confidence", size="pop_density", hover_name="area",
            opacity=0.85, size_max=20,
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            labels={"med_income_aed":"Income","footfall_daily":"Footfall","composite_score":"Score"},
            title="3D: Income × Footfall × Score", height=550
        )
        fig_3d.update_layout(**CHART_BG)
        st.plotly_chart(fig_3d, use_container_width=True)
        inference("<b>Green dots at top-right-front = best opportunities</b> — "
                  "high income, high footfall, high score. Rotate with mouse to explore clusters.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📏 Parallel Coordinates")
            sample = df_f.sample(min(300, len(df_f)), random_state=42)
            fig_par = px.parallel_coordinates(
                sample, dimensions=FEATURES, color="confidence",
                color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                title="All 7 Features vs Confidence", height=450
            )
            fig_par.update_layout(**CHART_BG)
            st.plotly_chart(fig_par, use_container_width=True)
            inference("<b>Green lines = best sites.</b> Drag any axis to filter. "
                      "Best sites show high income, high footfall, low competitors.")

        with col2:
            st.markdown("### 🔥 Feature Correlation Heatmap")
            corr = df_f[FEATURES + ["composite_score"]].corr()
            fig_heat = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale=[[0,"#764ba2"],[0.5,"#1e1b4b"],[1,"#22c55e"]],
                title="Feature Correlations", height=450
            )
            fig_heat.update_layout(**CHART_BG)
            st.plotly_chart(fig_heat, use_container_width=True)
            inference("<b>Green (+1) = strong positive driver.</b> Competitors showing negative "
                      "correlation confirms more rivals = lower site attractiveness.")

        st.markdown("### 🎬 Animated Bubble Chart")
        df_anim = df_f.copy()
        df_anim["income_band"] = pd.cut(df_f["med_income_aed"], bins=3,
                                         labels=["Low Income","Mid Income","High Income"])
        fig_anim = px.scatter(
            df_anim, x="med_income_aed", y="footfall_daily",
            animation_frame="income_band", size="composite_score", color="confidence",
            hover_name="area", size_max=40,
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            title="Sites Animated by Income Band", height=480
        )
        fig_anim.update_layout(**CHART_BG)
        st.plotly_chart(fig_anim, use_container_width=True)
        inference("Press ▶ to animate through income bands. <b>More green bubbles appear in "
                  "the High Income frame</b> — wealthier neighbourhoods = better Lenskart prospects.")

        st.markdown("### 🎛️ What Drives a Recommended Site?")
        imp_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Importance": rf.feature_importances_
        }).sort_values("Importance")
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="viridis",
                         title="Feature Importance (Random Forest)", height=380)
        fig_imp.update_layout(**CHART_BG, xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_imp, use_container_width=True)
        inference("<b>Longer bar = bigger impact on success.</b> Focus scouting efforts "
                  "on the top 2–3 factors — they determine 80% of a site's outcome.")

    with tab2:
        st.markdown("### 🧬 Location Type Profiles")
        col1, col2 = st.columns(2)
        with col1:
            cluster_means = df_f.groupby("location_type")[FEATURES].mean()
            feat_labels = list(FEATURE_LABELS.values())
            fig_radar = go.Figure()
            for i, (lt, row) in enumerate(cluster_means.iterrows()):
                vals = row.tolist()
                mn, mx = min(vals), max(vals) + 1e-9
                norm = [(v-mn)/(mx-mn) for v in vals]
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm+[norm[0]], theta=feat_labels+[feat_labels[0]],
                    fill="toself", name=lt, line_color=COLORS[i % len(COLORS)]
                ))
            fig_radar.update_layout(
                **CHART_BG, title="Location Type Radar", height=470,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            inference("<b>Wider shape = stronger across all factors.</b> "
                      "Premium Hub and High Footfall types filling most of the radar are your best zones.")

        with col2:
            fig_violin = px.violin(
                df_f, y="confidence", x="location_type", box=True,
                points="outliers", color="location_type",
                title="Confidence Distribution by Location Type", height=470
            )
            fig_violin.update_layout(**CHART_BG, xaxis=AXIS, yaxis=AXIS, showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)
            inference("<b>Tall narrow violin near the top = consistently high confidence.</b> "
                      "Wide violins = unpredictable zone — some great sites, some terrible. "
                      "White dot = median confidence.")

        st.markdown("### 🔗 Key Insights (Association Rules)")
        df_bin = df_f[["near_mall","metro_access","optimal_site"]].astype(bool)
        freq = apriori(df_bin, min_support=0.1, use_colnames=True)
        if not freq.empty:
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            rules = rules.rename(columns={
                "antecedents": "IF (Antecedent)", "consequents": "THEN (Consequent)",
                "support": "Support", "confidence": "Confidence", "lift": "Lift"
            })
            rules["IF (Antecedent)"]   = rules["IF (Antecedent)"].apply(lambda x: " + ".join(list(x)))
            rules["THEN (Consequent)"] = rules["THEN (Consequent)"].apply(lambda x: " + ".join(list(x)))
            rules = rules.round(3)

            c1, c2, c3 = st.columns(3)
            c1.metric("Rules Found",    len(rules))
            c2.metric("Max Lift",       f"{rules['Lift'].max():.2f}x")
            c3.metric("Max Confidence", f"{rules['Confidence'].max():.1%}")

            st.dataframe(
                rules[["IF (Antecedent)","THEN (Consequent)","Support","Confidence","Lift"]].head(10),
                use_container_width=True
            )
            inference("IF near_mall + metro → THEN optimal_site reads as a business rule. "
                      "<b>Lift > 1 = the combination is more powerful than chance.</b> "
                      "Use these as quick checklists when scouting new locations.")

            fig_rules = px.scatter(
                rules, x="Support", y="Confidence", size="Lift", color="Lift",
                hover_data={"IF (Antecedent)": True, "THEN (Consequent)": True},
                color_continuous_scale="plasma",
                title="Support vs Confidence (Bubble Size = Lift)", height=380
            )
            fig_rules.update_layout(**CHART_BG, xaxis=AXIS, yaxis=AXIS)
            st.plotly_chart(fig_rules, use_container_width=True)
            inference("<b>Top-right large bubbles = most actionable rules</b> — "
                      "high support, high confidence, and high lift all together.")

    with tab3:
        st.markdown("### ⏳ 3-Year Risk Aging Matrix")
        risk_data = {"Low":[1.0,1.1,1.2],"Medium":[1.0,1.4,1.9],"High":[1.0,1.8,2.8]}
        counts = df_f["risk_level"].value_counts()
        rows = []
        for rl, mults in risk_data.items():
            if rl in counts.index:
                for yr, m in zip([1,2,3], mults):
                    rows.append({"Risk Level":rl,"Year":f"Year {yr}","Risk Score":round(counts[rl]*m)})
        aging_df = pd.DataFrame(rows)

        col1, col2 = st.columns(2)
        with col1:
            fig_line = px.line(aging_df, x="Year", y="Risk Score", color="Risk Level",
                               markers=True,
                               color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"},
                               title="Risk Evolution Over 3-Year Lease", height=400)
            fig_line.update_traces(line_width=3, marker_size=10)
            fig_line.update_layout(**CHART_BG, xaxis=AXIS, yaxis=AXIS)
            st.plotly_chart(fig_line, use_container_width=True)
            inference("<b>Low-risk sites stay flat. High-risk compounds to 2.8× by Year 3.</b> "
                      "Always sign shorter leases on medium/high-risk sites.")

        with col2:
            fig_area = px.area(aging_df, x="Year", y="Risk Score", color="Risk Level",
                               color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"},
                               title="Cumulative Risk Exposure", height=400)
            fig_area.update_layout(**CHART_BG, xaxis=AXIS, yaxis=AXIS)
            st.plotly_chart(fig_area, use_container_width=True)
            inference("<b>Growing red area = portfolio ageing badly.</b> "
                      "Urgent signal to renegotiate or exit high-risk leases before Year 3.")

        st.markdown("### 🔥 Area × Risk Heatmap")
        heat_pivot = df_f.groupby(["area","risk_level"]).size().unstack(fill_value=0)
        fig_heat2 = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r",
                              text_auto=True, aspect="auto",
                              title="Number of Sites per Area by Risk Level", height=420)
        fig_heat2.update_layout(**CHART_BG)
        st.plotly_chart(fig_heat2, use_container_width=True)
        inference("<b>Dark red = many high-risk sites — avoid. Dark green = safe expansion zone.</b> "
                  "Areas green across the full row are your safest Dubai territories.")

        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Analysis CSV", csv,
                           "lenskart_analysis.csv", "text/csv", use_container_width=True)


# ════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ════════════════════════════════════════════════
elif page == "🎯 Predict Location":

    st.markdown("## 🎯 Location Predictor")
    st.caption("Enter site details and get an instant AI-powered recommendation")

    col_form, col_result = st.columns([1,1])

    with col_form:
        st.markdown('''<div class="hero-card">''', unsafe_allow_html=True)
        st.markdown("### 📝 Site Details")
        area_input  = st.text_input("📍 Area Name (optional)", "JLT")
        income      = st.slider("💰 Median Income (AED/month)",  7500, 45000, 18000, 500)
        footfall    = st.slider("👣 Daily Footfall Estimate",      200, 45000,  8000, 100)
        density     = st.slider("👥 Population Density",           1.0,  45.0,  15.0,   0.5)
        age         = st.slider("👶 Median Age",                    20,    55,    32)
        competitors = st.slider("🏪 Nearby Competitors",             0,    10,     2)
        near_mall   = st.toggle("🏬 Near a Mall?",   True)
        metro       = st.toggle("🚇 Metro Access?",  True)
        st.markdown('''</div>''', unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict This Location", use_container_width=True)

    with col_result:
        if predict_btn:
            new_X      = np.array([[density, age, income, competitors, footfall,
                                    int(near_mall), int(metro)]])
            new_scaled = scaler.transform(new_X)
            cluster_id = int(kmeans.predict(new_scaled)[0])
            prob       = float(rf.predict_proba(new_scaled)[0][1])
            loc_type   = CLUSTER_NAMES.get(cluster_id, "Mixed Zone")
            stars      = ("⭐⭐⭐⭐⭐" if prob > 0.80 else
                          "⭐⭐⭐⭐"   if prob > 0.65 else
                          "⭐⭐⭐"     if prob > 0.40 else "⭐⭐")

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
                {badge}<br><br>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;text-align:left;margin-top:1rem">
                    <div>📊 Confidence<br><b style="font-size:1.5rem">{prob:.1%}</b></div>
                    <div>🧬 Zone Type<br><b style="font-size:1.1rem">{loc_type}</b></div>
                </div>
            </div>""", unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob*100, 1),
                delta={"reference": 65, "valueformat": ".1f"},
                title={"text": "Confidence Score (%)", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white", "size": 48}},
                gauge={
                    "axis": {"range": [0,100], "tickcolor": "white"},
                    "bar":  {"color": "#667eea", "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "bordercolor": "rgba(255,255,255,0.2)",
                    "steps": [
                        {"range":[0,40],  "color":"rgba(239,68,68,0.3)"},
                        {"range":[40,65], "color":"rgba(245,158,11,0.3)"},
                        {"range":[65,100],"color":"rgba(34,197,94,0.3)"}
                    ],
                    "threshold": {"line":{"color":"white","width":3},"thickness":0.8,"value":65}
                }
            ))
            fig_gauge.update_layout(**CHART_BG, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)
            inference("<b>Green zone (65–100%) = GO. Yellow = Review. Red = Avoid.</b> "
                      "The white line at 65% is Lenskart's minimum confidence threshold for lease commitment.")

            st.markdown("### 🔍 Score Breakdown")
            components = {
                "Footfall":           0.30 * float(np.log1p(footfall)) / 10,
                "Demographics":       0.25 * (1 if age < 38 and income > 15000 else 0),
                "Low Competition":    0.20 * (1 / (1 + competitors)),
                "Accessibility":      0.15 * (int(near_mall) + int(metro)),
                "Population Density": 0.10 * (density / 20)
            }
            sc_df       = pd.DataFrame(components.items(), columns=["Factor","Score"])
            total_score = sc_df["Score"].sum()
            fig_wf = go.Figure(go.Waterfall(
                name="Score", orientation="v",
                x=sc_df["Factor"].tolist() + ["Total"],
                y=sc_df["Score"].tolist() + [None],
                measure=["relative"]*len(sc_df) + ["total"],
                connector={"line":{"color":"rgba(255,255,255,0.3)"}},
                increasing={"marker":{"color":"#22c55e"}},
                totals={"marker":    {"color":"#667eea"}},
                text=[f"{v:.3f}" for v in sc_df["Score"]] + [f"{total_score:.3f}"]
            ))
            fig_wf.update_layout(**CHART_BG, xaxis=AXIS, yaxis=AXIS,
                                 title="Composite Score Decomposition", height=380)
            st.plotly_chart(fig_wf, use_container_width=True)
            inference("<b>Taller green bar = that factor is working in your favour.</b> "
                      "Demographics = 0 means the area's profile doesn't match Lenskart's target customer. "
                      "The blue Total bar must exceed 0.65 for a GO decision.")
        else:
            st.markdown("""
            <div class="hero-card" style="text-align:center;padding:4rem 2rem">
                <div style="font-size:5rem">🎯</div>
                <h3>Ready to Predict!</h3>
                <p style="color:rgba(255,255,255,0.6)">Fill in site details on the left<br>
                and hit <b>Predict This Location</b></p>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# PAGE 4 — AI CHATBOT
# ════════════════════════════════════════════════
elif page == "🤖 AI Assistant":

    st.markdown("""
    <div class="hero-card">
        <h1 style="font-size:2.2rem;margin:0">🤖 Lenskart AI Assistant</h1>
        <p style="font-size:1rem;color:rgba(255,255,255,0.7);margin-top:0.5rem">
        Ask me anything about Dubai site selection, ML results, risks, or how to use this dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggested prompts
    st.markdown("**💬 Suggested Questions:**")
    sg1, sg2, sg3, sg4 = st.columns(4)
    prompts = {
        "🏆 Top sites":           "What are the top recommended sites?",
        "⚠️ Risk analysis":       "Give me a risk analysis",
        "🎛️ Key drivers":         "What factors drive a successful outlet?",
        "📊 Dashboard summary":   "Give me an overview of the data"
    }
    triggered = None
    for col, (label, prompt) in zip([sg1, sg2, sg3, sg4], prompts.items()):
        with col:
            if st.button(label, key=f"btn_{label}", use_container_width=True):
                triggered = prompt

    st.markdown("---")

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content":
             "👋 Hi! I'm your **Lenskart Dubai AI Assistant**. I have full access to "
             "the site analysis data and ML results.\n\n"
             "Ask me about top locations, risk levels, what drives success, "
             "or how to interpret any chart on this dashboard!"}
        ]

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"],
                             avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Handle suggested prompt button click
    if triggered:
        st.session_state.chat_history.append({"role": "user", "content": triggered})
        response = chatbot_response(triggered, df_f)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask me anything about Lenskart's Dubai expansion... 🕶️"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = chatbot_response(user_input, df_f)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear button
    if len(st.session_state.chat_history) > 1:
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = [st.session_state.chat_history[0]]
            st.rerun()


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem;color:rgba(255,255,255,0.4);font-size:0.85rem">
    🕶️ Lenskart Dubai Intelligence · ML-Powered · Production Ready
</div>
""", unsafe_allow_html=True)
