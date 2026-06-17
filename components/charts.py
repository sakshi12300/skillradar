"""Plotly chart builders for SkillRadar."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
TEXT      = "#c9d1d9"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"

CATEGORY_COLORS = {
    "Language": "#58a6ff",
    "Data Eng": "#3fb950",
    "AI/ML":    "#d2a8ff",
    "DevOps":   "#ffa657",
    "Cloud":    "#79c0ff",
    "BI":       "#ffd700",
    "Other":    "#8b949e",
}

_layout = dict(
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
    font=dict(color=TEXT, size=12),
    margin=dict(l=0, r=0, t=36, b=0),
)


# ── Skill trend lines ──────────────────────────────────────────────────────

def skill_trend_chart(demand: pd.DataFrame, selected_skills: list[str]) -> go.Figure:
    filtered = demand[demand["skill"].isin(selected_skills)]
    fig = px.line(
        filtered, x="month_str", y="demand_rate", color="skill",
        labels={"month_str": "", "demand_rate": "Demand Rate", "skill": "Skill"},
        title="Skill Demand Over Time",
    )
    fig.update_traces(line=dict(width=2.2))
    fig.update_layout(**_layout, legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(tickformat=".0%")
    return fig


# ── Rising vs falling ──────────────────────────────────────────────────────

def skill_momentum_chart(half_life: pd.DataFrame) -> go.Figure:
    top_rising  = half_life[half_life["status"] == "growing"].nlargest(8, "avg_monthly_delta")
    top_falling = half_life[half_life["status"] == "declining"].nsmallest(8, "avg_monthly_delta")
    combined    = pd.concat([top_rising, top_falling])
    combined    = combined.sort_values("avg_monthly_delta", ascending=True)

    colors = [GREEN if v >= 0 else RED for v in combined["avg_monthly_delta"]]
    fig = go.Figure(go.Bar(
        x=combined["avg_monthly_delta"] * 100,
        y=combined["skill"],
        orientation="h",
        marker_color=colors,
        text=combined["avg_monthly_delta"].apply(lambda v: f"{v*100:+.2f}%/mo"),
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    fig.update_layout(
        **_layout,
        title="Skill Momentum (Monthly Demand Δ)",
        xaxis_title="Monthly Change (%)",
        yaxis_title=None,
    )
    return fig


# ── Skill half-life table chart ────────────────────────────────────────────

def half_life_bar(half_life: pd.DataFrame) -> go.Figure:
    declining = half_life[half_life["half_life_months"].notna()].nsmallest(10, "half_life_months")
    fig = go.Figure(go.Bar(
        x=declining["half_life_months"],
        y=declining["skill"],
        orientation="h",
        marker_color=RED,
        text=declining["half_life_months"].apply(lambda v: f"{v:.0f} mo"),
        textposition="outside",
    ))
    fig.update_layout(
        **_layout,
        title="Skill Half-Life — Fastest Declining",
        xaxis_title="Months until demand halves",
        yaxis_title=None,
    )
    return fig


# ── Role DNA scatter ───────────────────────────────────────────────────────

def role_dna_scatter(pca_df: pd.DataFrame, cluster_labels: dict[int, str]) -> go.Figure:
    pca_df = pca_df.copy()
    pca_df["cluster_name"] = pca_df["cluster_id"].map(cluster_labels)
    fig = px.scatter(
        pca_df, x="pca_x", y="pca_y",
        color="cluster_name",
        symbol="role_title",
        hover_data=["role_title"],
        title="Role DNA Map — Skill-Space Clustering",
        labels={"pca_x": "Component 1", "pca_y": "Component 2",
                "cluster_name": "Cluster", "role_title": "Role"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.update_layout(**_layout, legend=dict(orientation="h", y=-0.15))
    return fig


# ── Role convergence heatmap ───────────────────────────────────────────────

def convergence_heatmap(conv: pd.DataFrame) -> go.Figure:
    roles  = sorted(set(conv["role_a"].tolist() + conv["role_b"].tolist()))
    matrix = pd.DataFrame(1.0, index=roles, columns=roles)
    for _, row in conv.iterrows():
        matrix.loc[row["role_a"], row["role_b"]] = row["similarity"]
        matrix.loc[row["role_b"], row["role_a"]] = row["similarity"]

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[[0, DARK_BG], [0.5, "#1f6feb"], [1, ACCENT]],
        zmin=0, zmax=1,
        text=matrix.round(2).values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        **_layout,
        title="Role Convergence — Skill Overlap Heatmap",
        xaxis=dict(tickangle=-30),
    )
    return fig


# ── Salary premium bar ─────────────────────────────────────────────────────

def salary_premium_chart(premium_df: pd.DataFrame) -> go.Figure:
    top = premium_df.head(12)
    colors = [GREEN if v >= 0 else RED for v in top["premium"]]
    fig = go.Figure(go.Bar(
        x=top["skill"], y=top["premium"],
        marker_color=colors,
        text=top["premium"].apply(lambda v: f"${v:+,}"),
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_layout(
        **_layout,
        title="Salary Premium by Skill (vs. Market Average)",
        yaxis_title="Premium ($)",
        xaxis_title=None,
    )
    return fig


# ── Transparency trend ─────────────────────────────────────────────────────

def transparency_chart(trans: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trans["month_str"], y=trans["transparency_rate"],
        mode="lines+markers",
        line=dict(color=YELLOW, width=2.2),
        fill="tozeroy", fillcolor=f"rgba(210,153,34,0.10)",
        name="Transparency Rate",
    ))
    fig.update_layout(
        **_layout,
        title="Salary Transparency Rate Over Time",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        xaxis_title=None,
    )
    return fig


# ── Salary by role ─────────────────────────────────────────────────────────

def salary_by_role_chart(salary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Median",
        x=salary_df["role_title"], y=salary_df["median"],
        marker_color=ACCENT,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(salary_df["p75"] - salary_df["median"]).tolist(),
            arrayminus=(salary_df["median"] - salary_df["p25"]).tolist(),
            color=TEXT, thickness=1.5,
        ),
        text=salary_df["median"].apply(lambda v: f"${v/1000:.0f}k"),
        textposition="outside",
    ))
    fig.update_layout(
        **_layout,
        title="Salary by Role (Median + IQR)",
        yaxis_title="Annual Salary ($)",
        xaxis=dict(tickangle=-20),
    )
    return fig


# ── Work mode donut ────────────────────────────────────────────────────────

def work_mode_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["work_mode"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.58,
        marker_colors=[ACCENT, GREEN, YELLOW],
        textinfo="label+percent",
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color=TEXT),
    )
    return fig


# ── Skill adjacency heatmap (top N skills) ─────────────────────────────────

def adjacency_heatmap(adj: pd.DataFrame, top_n: int = 12) -> go.Figure:
    top_skills = (
        pd.concat([adj["skill_a"], adj["skill_b"]])
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    filtered = adj[adj["skill_a"].isin(top_skills) & adj["skill_b"].isin(top_skills)]
    matrix   = pd.DataFrame(0, index=top_skills, columns=top_skills)
    for _, row in filtered.iterrows():
        matrix.loc[row["skill_a"], row["skill_b"]] = row["co_count"]
        matrix.loc[row["skill_b"], row["skill_a"]] = row["co_count"]

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[[0, DARK_BG], [1, ACCENT]],
        text=matrix.values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        **_layout,
        title="Skill Co-Occurrence Matrix",
        xaxis=dict(tickangle=-30),
    )
    return fig
