"""
Synthetic job posting generator.
Simulates 18 months of real market patterns:
  - Python/AI skills surging; Perl/COBOL decaying
  - Data Engineer / ML Engineer roles converging in skill requirements
  - Salary language shifting toward "competitive" (less transparent over time)
  - Remote-first language rising then plateauing
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Skill universe ─────────────────────────────────────────────────────────
# Each entry: (skill_name, base_demand, monthly_growth_rate, category)
SKILLS = [
    # Rising fast
    ("Python",          0.72, +0.008, "Language"),
    ("SQL",             0.80, +0.003, "Language"),
    ("Spark",           0.38, +0.012, "Data Eng"),
    ("dbt",             0.18, +0.025, "Data Eng"),
    ("Airflow",         0.28, +0.018, "Data Eng"),
    ("Kafka",           0.22, +0.015, "Data Eng"),
    ("LLM",             0.05, +0.045, "AI/ML"),
    ("PyTorch",         0.24, +0.020, "AI/ML"),
    ("MLflow",          0.12, +0.022, "AI/ML"),
    ("Terraform",       0.20, +0.014, "DevOps"),
    ("Docker",          0.55, +0.006, "DevOps"),
    ("Kubernetes",      0.32, +0.010, "DevOps"),
    ("Snowflake",       0.30, +0.018, "Data Eng"),
    ("Databricks",      0.20, +0.020, "Data Eng"),
    ("DuckDB",          0.04, +0.035, "Data Eng"),
    # Stable
    ("AWS",             0.68, +0.002, "Cloud"),
    ("GCP",             0.38, +0.004, "Cloud"),
    ("Azure",           0.42, +0.003, "Cloud"),
    ("Tableau",         0.35, +0.001, "BI"),
    ("Power BI",        0.30, +0.002, "BI"),
    ("scikit-learn",    0.40, +0.003, "AI/ML"),
    ("pandas",          0.60, +0.002, "Language"),
    ("Java",            0.45, -0.002, "Language"),
    ("Scala",           0.20, -0.003, "Language"),
    # Decaying
    ("R",               0.28, -0.008, "Language"),
    ("Hadoop",          0.18, -0.015, "Data Eng"),
    ("Hive",            0.15, -0.014, "Data Eng"),
    ("SAS",             0.12, -0.012, "BI"),
    ("Excel",           0.50, -0.006, "BI"),
    ("Oracle",          0.22, -0.010, "Language"),
    ("Perl",            0.05, -0.020, "Language"),
]

# ── Role templates ─────────────────────────────────────────────────────────
# Each role: (title, core_skills, optional_skills, base_salary_mid)
ROLES = {
    "Data Engineer":      (["Python","SQL","Spark","Airflow","Kafka"],      ["dbt","Snowflake","Databricks","Docker","AWS"],      130_000),
    "ML Engineer":        (["Python","PyTorch","Docker","Kubernetes","AWS"], ["MLflow","Spark","Kafka","Terraform","Databricks"],  145_000),
    "Data Scientist":     (["Python","SQL","scikit-learn","pandas"],         ["PyTorch","Spark","Tableau","AWS","MLflow"],          125_000),
    "Analytics Engineer": (["SQL","dbt","Python","Snowflake"],               ["Tableau","Power BI","Airflow","GCP"],               115_000),
    "Data Analyst":       (["SQL","Excel","Tableau","Python"],               ["Power BI","pandas","AWS","Power BI"],               90_000),
    "LLM Engineer":       (["Python","LLM","PyTorch","Docker"],              ["Kubernetes","AWS","MLflow","Kafka"],                155_000),
    "Platform Engineer":  (["Terraform","Kubernetes","Docker","AWS"],        ["GCP","Azure","Kafka","Spark"],                      140_000),
    "BI Developer":       (["SQL","Tableau","Power BI","Excel"],             ["Python","Snowflake","dbt","Azure"],                 100_000),
}

COMPANIES = [
    "Stripe", "Databricks", "Snowflake", "Airbnb", "Uber", "Netflix",
    "Meta", "Google", "Amazon", "Microsoft", "Figma", "Notion",
    "Vercel", "Anthropic", "OpenAI", "Palantir", "Confluent", "dbt Labs",
    "Aiven", "Prefect", "Monte Carlo", "Great Expectations", "Fivetran",
]

WORK_MODES = ["Remote", "Hybrid", "Onsite"]

SALARY_LANGUAGE_TRANSPARENT = [
    "Salary: ${low}k–${high}k",
    "Compensation: ${low},000–${high},000/yr",
    "${low}k–${high}k base + equity",
]
SALARY_LANGUAGE_OPAQUE = [
    "Competitive compensation package",
    "Market-rate salary + meaningful equity",
    "Top-of-market comp for exceptional candidates",
    "Compensation discussed during interview",
]

JOB_DESCRIPTION_TEMPLATES = [
    "We are looking for a {title} to join our {team} team. You will {action1} and {action2}. "
    "Strong experience with {skill1}, {skill2}, and {skill3} is required. "
    "Nice to have: {skill4}, {skill5}. {salary_text}. {work_mode} position.",

    "Join our {team} team as a {title}. "
    "You'll own {action1}, collaborate on {action2}, and drive {action3}. "
    "Must have: {skill1}, {skill2}. Preferred: {skill3}, {skill4}. {salary_text}.",

    "As a {title} at {company}, you'll {action1}. "
    "We expect expertise in {skill1} and {skill2}, with hands-on {skill3} experience. "
    "{salary_text}. We are a {work_mode} company.",
]

ACTIONS = [
    "build and maintain scalable data pipelines",
    "design real-time streaming architectures",
    "deploy and monitor ML models in production",
    "develop self-serve analytics infrastructure",
    "own the data platform end-to-end",
    "collaborate cross-functionally with product and engineering",
    "lead data quality and observability initiatives",
    "implement feature stores and ML pipelines",
    "architect cloud-native data solutions",
    "drive data democratisation across the organisation",
]

TEAMS = ["Data Platform", "ML Platform", "Analytics", "Growth", "Product Intelligence",
         "Infra", "AI Research", "Revenue Operations", "Core Engineering"]


def _skill_demand_at_month(skill_name: str, month: int) -> float:
    for name, base, growth, _ in SKILLS:
        if name == skill_name:
            return min(0.95, max(0.02, base + growth * month))
    return 0.0


def generate_job_postings(months: int = 18, n_records: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end   = datetime.now()
    start = end - timedelta(days=months * 30)

    records = []
    for _ in range(n_records):
        # Random timestamp uniformly across window
        offset_days = int(rng.integers(0, months * 30))
        ts          = start + timedelta(days=offset_days)
        month_idx   = offset_days // 30

        role_title  = rng.choice(list(ROLES.keys()))
        core_skills, opt_skills, base_salary = ROLES[role_title]
        company     = rng.choice(COMPANIES)
        work_mode   = rng.choice(WORK_MODES, p=[0.45, 0.40, 0.15])

        # Remote trend: probability of remote rises over time
        if rng.random() < 0.02 * (month_idx / months):
            work_mode = "Remote"

        # Select skills visible in this posting (demand-weighted)
        all_pool = core_skills + opt_skills
        visible  = []
        for sk in all_pool:
            prob = _skill_demand_at_month(sk, month_idx)
            if rng.random() < prob:
                visible.append(sk)
        if not visible:
            visible = core_skills[:2]

        # Salary transparency: decreases over time (market trend)
        salary_sd   = int(rng.integers(5, 20)) * 1_000
        salary_low  = max(60_000, base_salary - salary_sd + int(rng.integers(-10_000, 10_000)))
        salary_high = salary_low + int(rng.integers(20_000, 60_000))

        # Transparency probability: starts 0.65, falls to 0.30
        transparent_prob = max(0.30, 0.65 - 0.02 * month_idx)
        if rng.random() < transparent_prob:
            tmpl = rng.choice(SALARY_LANGUAGE_TRANSPARENT)
            salary_text = tmpl.replace("{low}", str(salary_low // 1000)).replace("{high}", str(salary_high // 1000))
            salary_disclosed = True
        else:
            salary_text      = rng.choice(SALARY_LANGUAGE_OPAQUE)
            salary_disclosed = False

        # Build job description text
        desc_tmpl = rng.choice(JOB_DESCRIPTION_TEMPLATES)
        desc = (
            desc_tmpl
            .replace("{title}",    role_title)
            .replace("{company}",  company)
            .replace("{team}",     rng.choice(TEAMS))
            .replace("{action1}",  rng.choice(ACTIONS))
            .replace("{action2}",  rng.choice(ACTIONS))
            .replace("{action3}",  rng.choice(ACTIONS))
            .replace("{skill1}",   visible[0] if len(visible) > 0 else "Python")
            .replace("{skill2}",   visible[1] if len(visible) > 1 else "SQL")
            .replace("{skill3}",   visible[2] if len(visible) > 2 else "AWS")
            .replace("{skill4}",   visible[3] if len(visible) > 3 else "Docker")
            .replace("{skill5}",   visible[4] if len(visible) > 4 else "Kubernetes")
            .replace("{salary_text}", salary_text)
            .replace("{work_mode}",   work_mode)
        )

        records.append({
            "timestamp":        ts,
            "month_idx":        month_idx,
            "role_title":       role_title,
            "company":          company,
            "work_mode":        work_mode,
            "description":      desc,
            "skills_mentioned": visible,
            "salary_low":       salary_low,
            "salary_high":      salary_high,
            "salary_mid":       (salary_low + salary_high) // 2,
            "salary_disclosed": salary_disclosed,
        })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df
