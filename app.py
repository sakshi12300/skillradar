"""
SkillRadar — Tech Talent Market Intelligence Platform
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.generator      import generate_job_postings, SKILLS, ROLES
from pipeline.skills     import explode_skills, monthly_demand, skill_velocity, skill_half_life, skill_adjacency
from pipeline.roles      import build_role_skill_matrix, cluster_roles, pca_embed, top_skills_per_cluster, role_convergence_score
from pipeline.salary     import transparency_trend, salary_by_role, salary_by_workmode, skill_salary_premium, description_tone_vs_salary
from pipeline.forecast   import forecast_skill_demand, skill_demand_summary
from pipeline.companies  import company_hiring_rank, top_skills_per_company, skill_recommender
from components.charts   import (
    skill_trend_chart, skill_momentum_chart, half_life_bar,
    role_dna_scatter, convergence_heatmap,
    salary_premium_chart, transparency_chart, salary_by_role_chart,
    work_mode_donut, adjacency_heatmap,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkillRadar", page_icon="🎯",
    layout="wide", initial_sidebar_state="expanded",
)

# ── Advanced CSS — GitHub-dark + glassmorphism ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a0e17 0%, #0d1117 60%, #0a0e17 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(13,17,23,0.97);
    border-right: 1px solid rgba(88,166,255,0.15);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(88,166,255,0.07) 0%, rgba(63,185,80,0.04) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 14px;
    padding: 18px 20px;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"]:hover {
    border-color: rgba(88,166,255,0.5);
    box-shadow: 0 0 20px rgba(88,166,255,0.12);
    transform: translateY(-1px);
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c9d1d9 !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(88,166,255,0.05);
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(88,166,255,0.12);
}
.stTabs [data-baseweb="tab"] {
    color: #484f58 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 8px 16px !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(31,111,235,0.35) !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 8px 20px;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(35,134,54,0.3);
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(35,134,54,0.45);
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(88,166,255,0.06) !important;
    border: 1px solid rgba(88,166,255,0.18) !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(88,166,255,0.12) !important;
    overflow: hidden;
}

hr { border-color: rgba(88,166,255,0.1) !important; }
h1,h2,h3,h4,h5,p,label,span { color: #c9d1d9 !important; }

[data-testid="stExpander"] {
    background: rgba(88,166,255,0.04);
    border: 1px solid rgba(88,166,255,0.12);
    border-radius: 10px;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #58a6ff !important;
    box-shadow: 0 0 8px rgba(88,166,255,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper components ─────────────────────────────────────────────────────────
def section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin: 8px 0 18px 0;">
        <h3 style="
            background: linear-gradient(90deg, #58a6ff, #79c0ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 1.1rem; font-weight: 700; margin: 0 0 3px 0;
        ">{icon} {title}</h3>
        {"<p style='color:#484f58;font-size:0.78rem;margin:0;'>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def skill_pill(skill: str, color: str = "#58a6ff", size: str = "0.78rem") -> str:
    return f"""<span style="
        background:{color}14; color:{color};
        border:1px solid {color}30;
        border-radius:20px; padding:3px 11px;
        font-size:{size}; font-weight:600;
        display:inline-block; margin:3px 3px 3px 0;
    ">{skill}</span>"""


def career_gap_card(skill: str, have: bool) -> str:
    color  = "#3fb950" if have else "#f85149"
    icon   = "✓" if have else "✗"
    bg     = "rgba(63,185,80,0.08)" if have else "rgba(248,81,73,0.08)"
    border = "rgba(63,185,80,0.25)" if have else "rgba(248,81,73,0.25)"
    return f"""<div style="
        background:{bg}; border:1px solid {border};
        border-radius:8px; padding:8px 14px;
        display:flex; align-items:center; gap:10px;
        margin-bottom:6px;
    ">
        <span style="color:{color};font-size:1rem;font-weight:700;">{icon}</span>
        <span style="color:#c9d1d9;font-size:0.85rem;">{skill}</span>
        <span style="margin-left:auto;color:{color};font-size:0.72rem;font-weight:600;">
            {"HAVE" if have else "MISSING"}
        </span>
    </div>"""


def company_card(row: pd.Series) -> str:
    remote_color = "#3fb950" if row["pct_remote"] >= 50 else ("#d29922" if row["pct_remote"] >= 25 else "#8b949e")
    return f"""
    <div style="
        background: rgba(88,166,255,0.05);
        border: 1px solid rgba(88,166,255,0.12);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
        display: flex; align-items: center; gap: 16px;
    ">
        <div style="
            width:40px; height:40px;
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            border-radius: 10px;
            display:flex; align-items:center; justify-content:center;
            color:white; font-weight:800; font-size:1rem;
            flex-shrink:0;
        ">{row['company'][0]}</div>
        <div style="flex:1;">
            <div style="color:#c9d1d9;font-weight:700;font-size:0.9rem;">{row['company']}</div>
            <div style="color:#8b949e;font-size:0.72rem;margin-top:2px;">{row['roles']}</div>
        </div>
        <div style="text-align:right;">
            <div style="color:#3fb950;font-weight:700;font-size:0.85rem;">${row['avg_salary']/1000:.0f}k avg</div>
            <div style="color:{remote_color};font-size:0.72rem;">{row['pct_remote']:.0f}% remote</div>
        </div>
        <div style="
            background:rgba(88,166,255,0.1); border-radius:20px;
            padding:2px 10px; color:#58a6ff; font-size:0.72rem; font-weight:600;
        ">{int(row['total_postings'])} postings</div>
    </div>"""


# ── Pipeline cache ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all(months: int, n: int):
    df        = generate_job_postings(months=months, n_records=n)
    skill_df  = explode_skills(df)
    demand    = monthly_demand(skill_df, df)
    demand    = skill_velocity(demand)
    half_life = skill_half_life(demand)
    adj       = skill_adjacency(skill_df)
    fcast     = forecast_skill_demand(demand, horizon_months=6)
    summary   = skill_demand_summary(demand)

    role_mat    = build_role_skill_matrix(skill_df)
    role_mat    = cluster_roles(role_mat)
    pca_df      = pca_embed(role_mat)
    cluster_top = top_skills_per_cluster(role_mat)
    convergence = role_convergence_score(role_mat)

    trans       = transparency_trend(df)
    sal_role    = salary_by_role(df)
    sal_mode    = salary_by_workmode(df)
    sal_premium = skill_salary_premium(df)
    tone_sal    = description_tone_vs_salary(df)

    companies   = company_hiring_rank(df)

    return dict(
        df=df, skill_df=skill_df, demand=demand, half_life=half_life,
        adj=adj, fcast=fcast, summary=summary,
        role_mat=role_mat, pca_df=pca_df, cluster_top=cluster_top,
        convergence=convergence, trans=trans, sal_role=sal_role,
        sal_mode=sal_mode, sal_premium=sal_premium, tone_sal=tone_sal,
        companies=companies,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <div style="font-size:2rem;">🎯</div>
        <div style="
            background: linear-gradient(90deg, #58a6ff, #79c0ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size:1.4rem; font-weight:800; letter-spacing:-0.5px;
        ">SkillRadar</div>
        <div style="color:#484f58;font-size:0.72rem;margin-top:2px;">Talent Market Intelligence</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    with st.spinner("Loading talent market data…"):
        months   = st.slider("Data window (months)", 6, 24, 18)
        n_posted = st.select_slider("Posting volume",
                                    options=[500, 1000, 2000, 5000],
                                    value=2000,
                                    format_func=lambda x: f"{x:,} postings")
        data = load_all(months, n_posted)

    df, demand, half_life = data["df"], data["demand"], data["half_life"]

    st.divider()
    all_roles = sorted(df["role_title"].unique())
    sel_roles = st.multiselect("Roles", all_roles, default=all_roles)
    sel_wm    = st.multiselect("Work mode", ["Remote","Hybrid","Onsite"],
                               default=["Remote","Hybrid","Onsite"])

    st.divider()
    # Hot skills widget
    st.markdown("<div style='color:#484f58;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;'>🔥 Hottest Skills Right Now</div>", unsafe_allow_html=True)
    hot = half_life[half_life["status"]=="growing"].nlargest(5, "avg_monthly_delta")
    for _, row in hot.iterrows():
        pct = min(100, max(5, int(row["current_demand"] * 100)))
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:0.75rem;color:#c9d1d9;font-weight:600;">{row['skill']}</span>
                <span style="font-size:0.72rem;color:#3fb950;font-weight:700;">+{row['avg_monthly_delta']*100:.2f}%/mo</span>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:5px;">
                <div style="width:{pct}%;background:linear-gradient(90deg,#3fb950,#56d364);height:5px;
                    border-radius:4px;box-shadow:0 0 8px rgba(63,185,80,0.4);"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<div style='color:#484f58;font-size:0.72rem;'>📋 {len(df):,} postings · {months}mo window</div>", unsafe_allow_html=True)


# ── Filter ────────────────────────────────────────────────────────────────────
df_f = df[df["role_title"].isin(sel_roles) & df["work_mode"].isin(sel_wm)]
if df_f.empty:
    st.warning("No postings match current filters.")
    st.stop()


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(88,166,255,0.10) 0%, rgba(63,185,80,0.06) 100%);
    border: 1px solid rgba(88,166,255,0.18);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative; overflow: hidden;
">
    <div style="
        position:absolute; top:-40px; right:-40px;
        width:200px; height:200px; border-radius:50%;
        background: radial-gradient(circle, rgba(88,166,255,0.12), transparent);
        pointer-events:none;
    "></div>
    <div style="
        font-size:1.9rem; font-weight:800;
        background: linear-gradient(90deg, #c9d1d9, #79c0ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px; margin-bottom:6px;
    ">🎯 SkillRadar</div>
    <div style="color:#8b949e; font-size:0.9rem; max-width:640px; line-height:1.6;">
        Tech Talent Market Intelligence — track skill demand curves, detect
        role convergence, compute skill half-lives, and find your career gaps
        before the market makes them obvious.
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Postings",          f"{len(df_f):,}")
k2.metric("Skills Tracked",          f"{df_f['skills_mentioned'].explode().nunique()}")
k3.metric("Median Salary",           f"${df_f['salary_mid'].median()/1000:.0f}k")
k4.metric("Remote Postings",         f"{(df_f['work_mode']=='Remote').mean()*100:.0f}%")
k5.metric("Salary Transparency",     f"{df_f['salary_disclosed'].mean()*100:.0f}%")
st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Skill Trends",
    "🔮 Forecast",
    "🔬 Skill Intel",
    "🧬 Role DNA",
    "💰 Salary",
    "🏢 Companies",
    "🧭 Career Gap",
])
t_trends, t_forecast, t_intel, t_dna, t_salary, t_companies, t_career = tabs


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SKILL TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with t_trends:
    section_header("📈", "Skill Demand Trends", "Historical demand rate per skill — select any combo to compare")
    all_skills    = sorted(demand["skill"].unique())
    default_picks = ["Python","dbt","LLM","Hadoop","R","Spark","DuckDB"]
    sel_skills    = st.multiselect("Compare skills", all_skills,
                                   default=[s for s in default_picks if s in all_skills])
    if sel_skills:
        st.plotly_chart(skill_trend_chart(demand, sel_skills), use_container_width=True)
    else:
        st.info("Select skills above to plot demand trends.")

    st.divider()
    mc1, mc2 = st.columns(2)
    with mc1:
        section_header("⚡", "Skill Momentum", "Monthly demand velocity — positive = accelerating")
        st.plotly_chart(skill_momentum_chart(half_life), use_container_width=True)
    with mc2:
        section_header("⏳", "Skill Half-Life", "Months until demand halves — flag skills to deprioritise")
        st.plotly_chart(half_life_bar(half_life), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with t_forecast:
    section_header("🔮", "6-Month Skill Demand Forecast",
                   "Linear projection per skill — dashed line = forecast")

    fc_skills = st.multiselect(
        "Skills to forecast",
        all_skills,
        default=[s for s in ["Python","dbt","LLM","Hadoop","R"] if s in all_skills],
        key="fc_sel",
    )

    if fc_skills:
        fcast_df = data["fcast"]
        filtered = fcast_df[fcast_df["skill"].isin(fc_skills)]

        fig_f = go.Figure()
        colors = ["#58a6ff","#3fb950","#d2a8ff","#ffa657","#f85149","#ffd700","#79c0ff"]
        for i, skill in enumerate(fc_skills):
            sd = filtered[filtered["skill"] == skill]
            hist = sd[~sd["is_forecast"]]
            proj = sd[sd["is_forecast"]]
            c = colors[i % len(colors)]
            fig_f.add_trace(go.Scatter(
                x=hist["month_str"], y=hist["demand_rate"],
                mode="lines", name=f"{skill} (historical)",
                line=dict(color=c, width=2),
            ))
            fig_f.add_trace(go.Scatter(
                x=proj["month_str"], y=proj["demand_rate"],
                mode="lines", name=f"{skill} (forecast)",
                line=dict(color=c, width=2, dash="dot"),
                showlegend=False,
            ))
        fig_f.update_layout(
            title="Demand Rate Forecast (next 6 months)",
            xaxis_title=None, yaxis_title="Demand Rate",
            yaxis=dict(tickformat=".0%"),
            plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
            font=dict(color="#8b949e"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_f, use_container_width=True)

    st.divider()
    section_header("📊", "Forecast Summary Table", "Current vs projected demand in 6 months")
    summary = data["summary"]
    summary_view = summary[summary["skill"].isin(fc_skills)] if fc_skills else summary

    def color_change(val):
        if isinstance(val, float):
            return f"color: {'#3fb950' if val > 0 else '#f85149'}"
        return ""

    st.dataframe(
        summary_view.rename(columns={
            "skill":"Skill","current":"Now","in_6mo":"In 6 Months",
            "change":"Δ Demand","pct_change":"Δ %",
        }).style
        .format({"Now":"{:.1%}","In 6 Months":"{:.1%}","Δ Demand":"{:+.3f}","Δ %":"{:+.1f}%"})
        .applymap(color_change, subset=["Δ Demand","Δ %"]),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SKILL INTEL
# ─────────────────────────────────────────────────────────────────────────────
with t_intel:
    section_header("🔬", "Skill Co-Occurrence Matrix",
                   "How often two skills appear in the same posting — reveals bundled demand")
    st.plotly_chart(adjacency_heatmap(data["adj"]), use_container_width=True)

    st.divider()
    ic1, ic2 = st.columns([3, 2])
    with ic1:
        section_header("📋", "Full Skill Metrics")
        cat_filter = st.selectbox("Category", ["All"] + sorted(half_life["category"].unique()))
        hl_view = half_life if cat_filter == "All" else half_life[half_life["category"] == cat_filter]
        st.dataframe(
            hl_view[["skill","category","current_demand","peak_demand",
                     "avg_monthly_delta","half_life_months","status"]]
            .rename(columns={
                "skill":"Skill","category":"Category",
                "current_demand":"Demand (now)","peak_demand":"Peak",
                "avg_monthly_delta":"Monthly Δ","half_life_months":"Half-Life (mo)","status":"Status",
            }).style.format({
                "Demand (now)":"{:.1%}","Peak":"{:.1%}","Monthly Δ":"{:+.4f}",
            }),
            use_container_width=True, hide_index=True,
        )
    with ic2:
        section_header("🏷️", "Skills by Category")
        cat_counts = half_life.groupby("category")["skill"].count().reset_index()
        cat_counts.columns = ["Category","Skills"]
        cat_growing = half_life[half_life["status"]=="growing"].groupby("category")["skill"].count()
        cat_counts["Growing"] = cat_counts["Category"].map(cat_growing).fillna(0).astype(int)
        cat_counts["Declining"] = cat_counts["Skills"] - cat_counts["Growing"]
        for _, row in cat_counts.iterrows():
            pct_grow = int(row["Growing"] / max(row["Skills"], 1) * 100)
            st.markdown(f"""
            <div style="background:rgba(88,166,255,0.05);border:1px solid rgba(88,166,255,0.1);
                border-radius:8px;padding:10px 14px;margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="color:#c9d1d9;font-weight:600;font-size:0.85rem;">{row['Category']}</span>
                    <span style="color:#8b949e;font-size:0.75rem;">{int(row['Skills'])} skills</span>
                </div>
                <div style="display:flex;gap:6px;">
                    <span style="color:#3fb950;font-size:0.72rem;">{int(row['Growing'])} rising</span>
                    <span style="color:#484f58;">·</span>
                    <span style="color:#f85149;font-size:0.72rem;">{int(row['Declining'])} falling</span>
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:4px;margin-top:6px;">
                    <div style="width:{pct_grow}%;background:#3fb950;height:4px;border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ROLE DNA
# ─────────────────────────────────────────────────────────────────────────────
with t_dna:
    section_header("🧬", "Role DNA Map",
                   "KMeans + PCA — clustered by actual skill fingerprint, not job title")
    cluster_labels = {
        cid: f"Cluster {cid}: {', '.join(skills[:3])}"
        for cid, skills in data["cluster_top"].items()
    }
    st.plotly_chart(role_dna_scatter(data["pca_df"], cluster_labels), use_container_width=True)

    st.divider()
    dc1, dc2 = st.columns([3, 2])
    with dc1:
        section_header("🔗", "Role Convergence Heatmap",
                       "Cosine similarity between role skill-vectors — scores > 0.7 = boundary erosion")
        st.plotly_chart(convergence_heatmap(data["convergence"]), use_container_width=True)
    with dc2:
        section_header("🔍", "Top Differentiating Skills per Cluster")
        for cid, skills in data["cluster_top"].items():
            with st.expander(f"Cluster {cid} — {', '.join(skills[:2])}"):
                st.markdown(
                    "".join(skill_pill(s) for s in skills),
                    unsafe_allow_html=True,
                )

        st.divider()
        section_header("📊", "Top Convergent Role Pairs")
        top_conv = data["convergence"].head(5)
        for _, row in top_conv.iterrows():
            sim = row["similarity"]
            bar_w = int(sim * 100)
            color = "#f85149" if sim >= 0.8 else ("#d29922" if sim >= 0.6 else "#3fb950")
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-size:0.75rem;color:#c9d1d9;">
                        {row['role_a']} ↔ {row['role_b']}
                    </span>
                    <span style="font-size:0.75rem;color:{color};font-weight:700;">{sim:.2f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:5px;">
                    <div style="width:{bar_w}%;background:{color};height:5px;border-radius:4px;
                        box-shadow:0 0 8px {color}60;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SALARY
# ─────────────────────────────────────────────────────────────────────────────
with t_salary:
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        section_header("💰", "Salary by Role", "Median + IQR for disclosed postings only")
        st.plotly_chart(salary_by_role_chart(data["sal_role"]), use_container_width=True)
    with sc2:
        section_header("🌍", "Work Mode Mix")
        st.plotly_chart(work_mode_donut(df_f), use_container_width=True)
        section_header("📋", "By Mode")
        wm = data["sal_mode"].copy()
        wm["median"] = wm["median"].apply(lambda v: f"${v/1000:.0f}k")
        st.dataframe(wm.rename(columns={"work_mode":"Mode","median":"Median","mean":"Mean","count":"N"}),
                     use_container_width=True, hide_index=True)

    st.divider()
    sp1, sp2 = st.columns(2)
    with sp1:
        section_header("🏆", "Skill Salary Premium", "Avg salary lift for postings requiring this skill")
        st.plotly_chart(salary_premium_chart(data["sal_premium"]), use_container_width=True)
    with sp2:
        section_header("📉", "Salary Transparency Over Time", "% of postings disclosing a salary range")
        st.plotly_chart(transparency_chart(data["trans"]), use_container_width=True)

    st.divider()
    section_header("🗣️", "Job Description Tone vs. Salary Tier",
                   "VADER sentiment on description text — do high-paying jobs write differently?")
    tone = data["tone_sal"].copy()
    tone.columns = ["Salary Tier","Mean Sentiment","Median Sentiment","Count"]

    t_cols = st.columns(len(tone))
    for i, (_, row) in enumerate(tone.iterrows()):
        color = {"high":"#3fb950","mid":"#d29922","low":"#f85149"}.get(row["Salary Tier"], "#8b949e")
        t_cols[i].markdown(f"""
        <div style="background:rgba(88,166,255,0.05);border:1px solid rgba(88,166,255,0.1);
            border-radius:12px;padding:20px;text-align:center;">
            <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;
                letter-spacing:0.07em;margin-bottom:10px;">{row['Salary Tier'].upper()} TIER</div>
            <div style="color:{color};font-size:1.4rem;font-weight:800;">{row['Mean Sentiment']:+.3f}</div>
            <div style="color:#484f58;font-size:0.72rem;margin-top:4px;">mean VADER score</div>
            <div style="color:#8b949e;font-size:0.75rem;margin-top:8px;">{int(row['Count'])} postings</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — COMPANIES
# ─────────────────────────────────────────────────────────────────────────────
with t_companies:
    section_header("🏢", "Company Hiring Intelligence", "Top companies by posting volume + salary + remote rate")

    cmp1, cmp2 = st.columns([3, 2])
    with cmp1:
        companies = data["companies"]
        for _, row in companies.head(10).iterrows():
            st.markdown(company_card(row), unsafe_allow_html=True)

    with cmp2:
        section_header("🔍", "Drill Into a Company")
        sel_company = st.selectbox("Company", sorted(df["company"].unique()))
        c_skills = top_skills_per_company(df, sel_company, top_n=10)
        c_postings = df[df["company"] == sel_company]

        if c_skills:
            st.markdown(f"<div style='margin-bottom:10px;'>{''.join(skill_pill(s) for s in c_skills)}</div>",
                        unsafe_allow_html=True)

        st.metric("Total Postings", len(c_postings))
        if c_postings["salary_disclosed"].any():
            st.metric("Avg Salary (disclosed)",
                      f"${c_postings[c_postings['salary_disclosed']]['salary_mid'].mean()/1000:.0f}k")
        mode_counts = c_postings["work_mode"].value_counts()
        for mode, cnt in mode_counts.items():
            pct = cnt / len(c_postings) * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                <span style="color:#8b949e;font-size:0.78rem;width:60px;">{mode}</span>
                <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:6px;">
                    <div style="width:{pct:.0f}%;background:#58a6ff;height:6px;border-radius:4px;"></div>
                </div>
                <span style="color:#c9d1d9;font-size:0.78rem;font-weight:600;">{pct:.0f}%</span>
            </div>""", unsafe_allow_html=True)

    st.divider()
    section_header("🔗", "Skill Recommender",
                   "Given skills you already know — what should you learn next based on co-occurrence?")
    known = st.multiselect("Your current skills", sorted(df["skills_mentioned"].explode().unique()),
                           default=["Python","SQL","pandas"])
    if known:
        recs = skill_recommender(data["adj"], known)
        if recs:
            r_cols = st.columns(min(len(recs), 6))
            for i, rec in enumerate(recs[:6]):
                with r_cols[i]:
                    strength = min(100, int(rec["co_occurrence"] / 20 * 100))
                    color = "#3fb950" if strength >= 60 else ("#d29922" if strength >= 30 else "#58a6ff")
                    st.markdown(f"""
                    <div style="background:rgba(88,166,255,0.06);border:1px solid rgba(88,166,255,0.15);
                        border-radius:10px;padding:14px;text-align:center;">
                        <div style="color:#c9d1d9;font-weight:700;font-size:0.9rem;margin-bottom:6px;">
                            {rec['skill']}
                        </div>
                        <div style="color:{color};font-size:0.75rem;font-weight:600;">
                            {rec['co_occurrence']} co-occurrences
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:4px;margin-top:8px;">
                            <div style="width:{strength}%;background:{color};height:4px;border-radius:4px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No recommendations — try adding more skills.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — CAREER GAP
# ─────────────────────────────────────────────────────────────────────────────
with t_career:
    section_header("🧭", "Career Gap Analyzer",
                   "Select your target role + current skills — see exactly what's missing")

    cc1, cc2 = st.columns([1, 2])
    with cc1:
        target_role = st.selectbox("Target role", list(ROLES.keys()))
        your_skills = st.multiselect(
            "Skills you have",
            sorted(df["skills_mentioned"].explode().unique()),
            default=["Python","SQL","pandas"],
        )

    core_required, optional_skills, _ = ROLES[target_role]
    all_required = list(dict.fromkeys(core_required + optional_skills))

    with cc2:
        section_header("📋", f"Gap Analysis — {target_role}")

        have     = [s for s in all_required if s in your_skills]
        missing  = [s for s in core_required if s not in your_skills]
        nice_to  = [s for s in optional_skills if s not in your_skills and s not in core_required]
        readiness = len(have) / max(len(all_required), 1) * 100

        # Readiness bar
        r_color = "#3fb950" if readiness >= 70 else ("#d29922" if readiness >= 40 else "#f85149")
        st.markdown(f"""
        <div style="background:rgba(88,166,255,0.06);border:1px solid rgba(88,166,255,0.15);
            border-radius:12px;padding:18px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <span style="color:#c9d1d9;font-weight:700;">Role Readiness</span>
                <span style="color:{r_color};font-size:1.4rem;font-weight:800;">{readiness:.0f}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:6px;height:10px;">
                <div style="width:{readiness:.0f}%;background:linear-gradient(90deg,{r_color},{r_color}99);
                    height:10px;border-radius:6px;box-shadow:0 0 10px {r_color}50;
                    transition:width 0.5s;"></div>
            </div>
            <div style="display:flex;gap:16px;margin-top:10px;">
                <span style="color:#3fb950;font-size:0.78rem;">✓ {len(have)} have</span>
                <span style="color:#f85149;font-size:0.78rem;">✗ {len(missing)} core missing</span>
                <span style="color:#d29922;font-size:0.78rem;">○ {len(nice_to)} optional missing</span>
            </div>
        </div>""", unsafe_allow_html=True)

        ga1, ga2, ga3 = st.columns(3)
        with ga1:
            st.markdown("<div style='color:#3fb950;font-weight:600;font-size:0.8rem;margin-bottom:8px;'>✓ YOU HAVE</div>", unsafe_allow_html=True)
            for s in have:
                st.markdown(career_gap_card(s, True), unsafe_allow_html=True)
        with ga2:
            st.markdown("<div style='color:#f85149;font-weight:600;font-size:0.8rem;margin-bottom:8px;'>✗ CORE MISSING</div>", unsafe_allow_html=True)
            for s in missing:
                st.markdown(career_gap_card(s, False), unsafe_allow_html=True)
            if not missing:
                st.success("All core skills covered!")
        with ga3:
            st.markdown("<div style='color:#d29922;font-weight:600;font-size:0.8rem;margin-bottom:8px;'>○ NICE TO HAVE</div>", unsafe_allow_html=True)
            for s in nice_to[:6]:
                st.markdown(f"""<div style="background:rgba(210,153,34,0.06);border:1px solid rgba(210,153,34,0.2);
                    border-radius:8px;padding:8px 14px;margin-bottom:6px;color:#c9d1d9;font-size:0.84rem;">
                    ○ {s}</div>""", unsafe_allow_html=True)

    st.divider()
    if missing:
        section_header("📈", "Learning Priority — Missing Skills Demand",
                       "Sort your gap by market demand to prioritise what to learn first")
        missing_demand = half_life[half_life["skill"].isin(missing)].copy()
        if not missing_demand.empty:
            missing_demand = missing_demand.sort_values("current_demand", ascending=False)
            for _, row in missing_demand.iterrows():
                pct   = int(row["current_demand"] * 100)
                delta = row["avg_monthly_delta"] * 100
                color = "#3fb950" if delta > 0 else "#f85149"
                st.markdown(f"""
                <div style="background:rgba(248,81,73,0.05);border:1px solid rgba(248,81,73,0.15);
                    border-radius:10px;padding:12px 16px;margin-bottom:8px;
                    display:flex;align-items:center;gap:16px;">
                    <div style="width:36px;height:36px;background:rgba(248,81,73,0.15);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;
                        color:#f85149;font-weight:800;font-size:0.8rem;">#{list(missing_demand['skill']).index(row['skill'])+1}</div>
                    <div style="flex:1;">
                        <div style="color:#c9d1d9;font-weight:700;font-size:0.9rem;">{row['skill']}</div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:5px;margin-top:5px;">
                            <div style="width:{pct}%;background:#f85149;height:5px;border-radius:4px;"></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#c9d1d9;font-weight:600;font-size:0.85rem;">{row['current_demand']:.1%} demand</div>
                        <div style="color:{color};font-size:0.72rem;font-weight:600;">{delta:+.2f}%/mo</div>
                    </div>
                </div>""", unsafe_allow_html=True)
