"""
Salary signal intelligence:
  - Transparency rate over time (are companies hiding comp?)
  - Salary band by role, skill, work mode
  - Salary premium per skill (marginal salary lift of adding a skill)
  - Language pattern analysis: VADER on job description tone vs salary tier
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def transparency_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly % of postings that disclosed a salary range."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")
    monthly = (
        df.groupby("month")
        .agg(
            total          =("salary_disclosed", "count"),
            disclosed      =("salary_disclosed", "sum"),
        )
        .reset_index()
    )
    monthly["transparency_rate"] = monthly["disclosed"] / monthly["total"]
    monthly["month_str"]         = monthly["month"].astype(str)
    return monthly


def salary_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """Median / p25 / p75 salary by role_title, disclosed postings only."""
    disclosed = df[df["salary_disclosed"]].copy()
    return (
        disclosed.groupby("role_title")["salary_mid"]
        .describe(percentiles=[0.25, 0.75])
        .round(0)
        .reset_index()
        .rename(columns={"25%": "p25", "75%": "p75", "50%": "median"})
        [["role_title", "median", "p25", "p75", "count"]]
        .sort_values("median", ascending=False)
    )


def salary_by_workmode(df: pd.DataFrame) -> pd.DataFrame:
    disclosed = df[df["salary_disclosed"]].copy()
    return (
        disclosed.groupby("work_mode")["salary_mid"]
        .agg(median="median", mean="mean", count="count")
        .round(0)
        .reset_index()
        .sort_values("median", ascending=False)
    )


def skill_salary_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each skill, computes the average salary_mid of postings that mention it
    vs. postings that don't. Premium = (with_skill_avg - without_skill_avg).
    Only uses disclosed postings.
    """
    disclosed = df[df["salary_disclosed"]].copy()
    global_avg = disclosed["salary_mid"].mean()

    rows = []
    all_skills: set[str] = set()
    for lst in disclosed["skills_mentioned"]:
        all_skills.update(lst)

    for skill in sorted(all_skills):
        with_skill    = disclosed[disclosed["skills_mentioned"].apply(lambda s: skill in s)]
        without_skill = disclosed[disclosed["skills_mentioned"].apply(lambda s: skill not in s)]

        if len(with_skill) < 10:
            continue

        premium = with_skill["salary_mid"].mean() - global_avg
        rows.append({
            "skill":         skill,
            "avg_with":      round(with_skill["salary_mid"].mean()),
            "avg_without":   round(without_skill["salary_mid"].mean()) if len(without_skill) > 0 else None,
            "premium":       round(premium),
            "postings":      len(with_skill),
        })

    return pd.DataFrame(rows).sort_values("premium", ascending=False)


def description_tone_vs_salary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs VADER on job description text. Tests whether more 'positive' or
    'urgent' language correlates with higher or lower salaries.
    """
    df = df.copy()
    df["desc_sentiment"] = df["description"].apply(
        lambda t: _vader.polarity_scores(t)["compound"]
    )
    df["salary_tier"] = pd.qcut(
        df["salary_mid"], q=3, labels=["low", "mid", "high"]
    )
    return (
        df.groupby("salary_tier")["desc_sentiment"]
        .agg(mean="mean", median="median", count="count")
        .round(4)
        .reset_index()
    )
