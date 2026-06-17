# 🎯 SkillRadar — Tech Talent Market Intelligence Platform

SkillRadar is a Python-based analytics dashboard that simulates and analyzes tech hiring demand using skill fingerprints instead of job titles.

It generates synthetic job postings, computes skill demand velocity and half-life, clusters roles by skill similarity, and highlights salary transparency signals.

## Features

- Skill momentum and half-life metrics for technology demand trends
- Role clustering and PCA visualization based on actual skill vectors
- Convergence scores that compare role similarity across titles
- Salary transparency and sentiment analysis on posting language
- Fully interactive Streamlit dashboard with Plotly charts

## Requirements

- Python 3.11+
- `pip`

Dependencies are listed in `requirements.txt`.

## Install

1. Create and activate a virtual environment.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Run

Launch the dashboard with:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Project Structure

- `app.py` — Streamlit dashboard entry point
- `requirements.txt` — Python dependencies
- `data/generator.py` — synthetic job posting generator
- `pipeline/skills.py` — skill demand, velocity, half-life, and adjacency metrics
- `pipeline/roles.py` — clustering, PCA projections, and convergence scoring
- `pipeline/salary.py` — salary transparency, premium analysis, and tone scoring
- `components/charts.py` — Plotly chart builder helpers

## Dashboard Overview

The dashboard includes tabs for:

- Skill Trends
- Skill Intelligence
- Role DNA
- Salary Signals
- Postings Explorer

## Stop the app

Press `Ctrl + C` in the terminal to stop the server, then run:

```powershell
deactivate
```

## Notes

This repository uses synthetic data generated locally, so no external data downloads or API keys are required.
