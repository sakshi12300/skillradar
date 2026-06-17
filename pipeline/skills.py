"""
Skill signal extraction and market metrics.

Explodes the skills_mentioned list into one row per (posting, skill),
then computes:
  - Monthly demand rate (% of postings mentioning skill)
  - Demand velocity (month-over-month change)
  - Skill half-life: months until demand halves from its peak
  - Skill adjacency matrix (co-occurrence counts)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from data.generator import SKILLS


# Skill metadata lookup
SKILL_META: dict[str, dict] = {
    name: {"base": base, "growth": growth, "category": cat}
    for name, base, growth, cat in SKILLS
}


def explode_skills(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (job_id, skill). Adds skill category."""
    df = df.copy()
    df["job_id"] = df.index
    exploded = df.explode("skills_mentioned").rename(columns={"skills_mentioned": "skill"})
    exploded["category"] = exploded["skill"].map(
        lambda s: SKILL_META.get(s, {}).get("category", "Other")
    )
    return exploded.dropna(subset=["skill"])


def monthly_demand(skill_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns monthly demand rate per skill:
      demand_rate = (postings mentioning skill) / (total postings that month)
    """
    skill_df = skill_df.copy()
    skill_df["month"] = pd.to_datetime(skill_df["timestamp"]).dt.to_period("M")
    raw_df   = raw_df.copy()
    raw_df["month"] = pd.to_datetime(raw_df["timestamp"]).dt.to_period("M")

    monthly_totals = raw_df.groupby("month").size().reset_index(name="total_postings")
    monthly_skill  = skill_df.groupby(["month", "skill"]).size().reset_index(name="mentions")

    merged = monthly_skill.merge(monthly_totals, on="month")
    merged["demand_rate"] = merged["mentions"] / merged["total_postings"]
    merged["month_str"]   = merged["month"].astype(str)

    return merged.sort_values(["skill", "month"])


def skill_velocity(demand: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Velocity = rolling slope of demand_rate over `window` months.
    Positive = accelerating. Negative = decaying.
    """
    out = []
    for skill, grp in demand.groupby("skill"):
        grp = grp.sort_values("month").copy()
        grp["velocity"] = grp["demand_rate"].diff(window).fillna(0) / window
        grp["trend"]    = grp["velocity"].apply(
            lambda v: "rising" if v > 0.003 else ("falling" if v < -0.003 else "stable")
        )
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def skill_half_life(demand: pd.DataFrame) -> pd.DataFrame:
    """
    For each skill, estimates months until demand halves from its peak.
    For declining skills: half_life = ln(0.5) / decay_rate.
    Rising skills get half_life = inf (labelled 'Sustained growth').
    """
    records = []
    for skill, grp in demand.groupby("skill"):
        grp        = grp.sort_values("month")
        peak       = grp["demand_rate"].max()
        avg_change = grp["demand_rate"].diff().mean()
        cat        = SKILL_META.get(skill, {}).get("category", "Other")

        if avg_change >= 0 or peak < 0.01:
            half_life_months = None
            status = "growing"
        else:
            # months to fall from peak to peak/2
            half_life_months = round(abs((peak / 2) / avg_change), 1)
            status = "declining"

        records.append({
            "skill":             skill,
            "category":          cat,
            "peak_demand":       round(peak, 4),
            "avg_monthly_delta": round(avg_change, 5),
            "half_life_months":  half_life_months,
            "status":            status,
            "current_demand":    round(grp["demand_rate"].iloc[-1], 4),
        })
    return pd.DataFrame(records).sort_values("current_demand", ascending=False)


def skill_adjacency(skill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Co-occurrence matrix: how often skill A and skill B appear in the same posting.
    Returns a tidy (skill_a, skill_b, co_count) dataframe.
    """
    job_skills = (
        skill_df.groupby("job_id")["skill"]
        .apply(list)
        .reset_index()
    )
    pairs: dict[tuple, int] = {}
    for _, row in job_skills.iterrows():
        for a, b in combinations(sorted(set(row["skill"])), 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1

    return pd.DataFrame(
        [(a, b, c) for (a, b), c in pairs.items()],
        columns=["skill_a", "skill_b", "co_count"],
    ).sort_values("co_count", ascending=False)
