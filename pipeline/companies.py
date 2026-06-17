"""
Company-level intelligence:
  - Hiring volume ranking
  - Avg salary by company
  - Top skills demanded per company
  - Remote-friendliness score
"""
from __future__ import annotations

import pandas as pd


def company_hiring_rank(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    disclosed = df.copy()
    agg = (
        disclosed.groupby("company")
        .agg(
            total_postings   =("role_title", "count"),
            avg_salary       =("salary_mid", "mean"),
            pct_remote       =("work_mode", lambda s: (s == "Remote").mean()),
            pct_disclosed    =("salary_disclosed", "mean"),
            roles            =("role_title", lambda s: ", ".join(s.value_counts().head(2).index)),
        )
        .reset_index()
    )
    agg["avg_salary"]    = agg["avg_salary"].round(0)
    agg["pct_remote"]    = (agg["pct_remote"] * 100).round(1)
    agg["pct_disclosed"] = (agg["pct_disclosed"] * 100).round(1)
    return agg.nlargest(top_n, "total_postings").reset_index(drop=True)


def top_skills_per_company(df: pd.DataFrame, company: str, top_n: int = 8) -> list[str]:
    subset = df[df["company"] == company]
    if subset.empty:
        return []
    all_skills: list[str] = []
    for lst in subset["skills_mentioned"]:
        all_skills.extend(lst)
    counts = pd.Series(all_skills).value_counts()
    return counts.head(top_n).index.tolist()


def skill_recommender(adjacency: pd.DataFrame, known_skills: list[str], top_n: int = 6) -> list[dict]:
    """
    Given skills you already know, recommends skills that co-occur most with
    your stack but aren't already in it.
    """
    related = adjacency[
        adjacency["skill_a"].isin(known_skills) | adjacency["skill_b"].isin(known_skills)
    ].copy()

    rec: dict[str, int] = {}
    for _, row in related.iterrows():
        other = row["skill_b"] if row["skill_a"] in known_skills else row["skill_a"]
        if other not in known_skills:
            rec[other] = rec.get(other, 0) + row["co_count"]

    return [
        {"skill": sk, "co_occurrence": cnt}
        for sk, cnt in sorted(rec.items(), key=lambda x: -x[1])[:top_n]
    ]
