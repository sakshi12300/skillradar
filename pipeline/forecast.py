"""
Skill demand forecasting using linear regression on monthly demand rates.
Projects 6 months forward per skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_skill_demand(demand: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    """
    For each skill, fits a linear trend on historical demand_rate and
    projects `horizon_months` forward.
    Returns a combined df with a 'is_forecast' flag.
    """
    out = []
    for skill, grp in demand.groupby("skill"):
        grp = grp.sort_values("month").copy()
        grp["is_forecast"] = False
        x = np.arange(len(grp))
        y = grp["demand_rate"].values

        if len(x) < 3:
            out.append(grp)
            continue

        coeffs    = np.polyfit(x, y, deg=1)
        slope, intercept = coeffs

        last_month = grp["month"].iloc[-1]
        future_months = pd.period_range(last_month + 1, periods=horizon_months, freq="M")
        future_x      = np.arange(len(grp), len(grp) + horizon_months)
        future_y      = np.clip(slope * future_x + intercept, 0, 1)

        forecast_df = pd.DataFrame({
            "month":       future_months,
            "month_str":   [str(m) for m in future_months],
            "skill":       skill,
            "demand_rate": future_y,
            "mentions":    0,
            "total_postings": 0,
            "velocity":    slope,
            "trend":       grp["trend"].iloc[-1] if "trend" in grp.columns else "stable",
            "is_forecast": True,
        })
        out.append(pd.concat([grp, forecast_df], ignore_index=True))

    return pd.concat(out, ignore_index=True)


def skill_demand_summary(demand: pd.DataFrame) -> pd.DataFrame:
    """Current demand + 6-month projected demand + expected change for top skills."""
    fcast = forecast_skill_demand(demand, horizon_months=6)

    rows = []
    for skill, grp in fcast.groupby("skill"):
        historical = grp[~grp["is_forecast"]]
        projected  = grp[grp["is_forecast"]]
        if historical.empty or projected.empty:
            continue
        current   = historical["demand_rate"].iloc[-1]
        projected6 = projected["demand_rate"].iloc[-1]
        rows.append({
            "skill":         skill,
            "current":       round(current, 4),
            "in_6mo":        round(projected6, 4),
            "change":        round(projected6 - current, 4),
            "pct_change":    round((projected6 - current) / max(current, 0.001) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("change", ascending=False)
