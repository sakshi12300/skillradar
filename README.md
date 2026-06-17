# 🎯 SkillRadar — Tech Talent Market Intelligence Platform

> **The job market is lying to you.** Job titles have never meant less.
> A "Data Engineer" at one company needs Spark and Kafka. At another, they just want Excel and SQL.
> SkillRadar ignores the labels and reads the actual skill fingerprints — tracking what the market is *really* hiring, what's rising, what's quietly dying, and where role boundaries are dissolving.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-KMeans%20%2B%20PCA%20%2B%20TF--IDF-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## The Problem No Dashboard Is Solving

Every "tech skills" report tells you Python is popular. That's not intelligence — that's noise.

The real questions are:

- **Which skills are accelerating?** dbt went from niche to mainstream in 18 months. What's next?
- **Which skills have a 6-month half-life?** Hadoop is dying. Hiring it costs you two years of dead learning.
- **Are Data Engineers and ML Engineers the same role now?** Their skill vectors say yes. Their job titles still disagree.
- **Is salary transparency increasing or eroding?** The answer should terrify you.
- **Does better pay come with better writing?** VADER on job description language vs. salary tier finds out.

SkillRadar answers all of it.

---

## What Makes This Unique

Most skill analytics tools count keyword frequencies. SkillRadar does five things nobody else does with basic Python:

| Feature | What it actually does |
|---|---|
| **Skill Half-Life** | Computes months until a skill's demand halves from its peak — flags what to stop learning now |
| **Skill Momentum** | Monthly demand velocity: rising vs. falling, with acceleration curves |
| **Role DNA Map** | KMeans + PCA on skill-presence vectors — clusters jobs by actual requirements, not job titles |
| **Convergence Score** | Cosine similarity between role skill-vectors — quantifies when two roles are becoming the same job |
| **Salary Transparency Index** | Tracks whether companies are hiding comp over time — and whether hiding it correlates with salary tier |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│   Job postings: 18 months × 5,000 records × 8 role types               │
│   Simulates real market patterns (LLM surge, Hadoop decay, dbt rise)   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │  SKILL PIPELINE  │ │  ROLE PIPELINE   │ │ SALARY PIPELINE  │
   │                  │ │                  │ │                  │
   │ • Demand rate    │ │ • Skill matrix   │ │ • Transparency   │
   │ • Velocity       │ │ • KMeans cluster │ │   trend          │
   │ • Half-life      │ │ • PCA 2D embed   │ │ • Role medians   │
   │ • Co-occurrence  │ │ • Convergence    │ │ • Skill premium  │
   │   adjacency      │ │   similarity     │ │ • Tone vs salary │
   └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
            └───────────────────┬┘                    │
                                ▼                     │
                   ┌────────────────────────────────┐ │
                   │     STREAMLIT DASHBOARD        │◄┘
                   │  5 tabs · 10 chart types       │
                   │  GitHub-dark theme             │
                   └────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why it's the right choice |
|---|---|---|
| Dashboard | Streamlit | Zero frontend code; full interactivity |
| Visualisation | Plotly | Scatter, heatmap, bar, line, donut — all interactive |
| Clustering | scikit-learn KMeans | Unsupervised role grouping with no labels |
| Dimensionality reduction | scikit-learn PCA | 2D skill-space projection for visual exploration |
| Skill extraction | pandas explode + pivot | Vectorised multi-label handling |
| Sentiment (tone analysis) | VADER | Job description language scoring |
| Data | pandas + NumPy | Vectorised pipeline across 5k postings |
| Language | Python 3.11+ | |

---

## Project Structure

```
skillradar/
│
├── app.py                     ← Streamlit dashboard (entry point)
├── requirements.txt
│
├── data/
│   └── generator.py           ← 18 months of synthetic job postings
│                                 with realistic skill demand curves
│
├── pipeline/
│   ├── skills.py              ← Demand rate, velocity, half-life, adjacency
│   ├── roles.py               ← KMeans clustering, PCA, convergence score
│   └── salary.py              ← Transparency trend, premium, tone analysis
│
└── components/
    └── charts.py              ← 10 Plotly chart builders
```

---

## How to Run It — Full Steps

### Step 0 — Check Python version

```bash
python --version     # needs 3.11+
# or
python3 --version
```

No Python? Get it at [python.org/downloads](https://www.python.org/downloads/).

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/yourname/skillradar.git
cd skillradar
```

---

### Step 2 — Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows — Command Prompt:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows — PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You'll see `(.venv)` in your prompt when it's active.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Installs: Streamlit, pandas, NumPy, Plotly, scikit-learn, VADER. ~60 seconds on first run.

---

### Step 4 — Launch

```bash
streamlit run app.py
```

You'll see:
```
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open [http://localhost:8501](http://localhost:8501). Dashboard loads in ~5 seconds as the ML pipeline runs.

> **No API keys. No data downloads.** The synthetic generator ships with the repo and produces 18 months of realistic skill demand data including the LLM surge, dbt rise, and Hadoop decay — all working out of the box.

---

### Step 5 — Explore

**To see skill half-life in action:** go to `Skill Trends` → `Skill Intelligence` tab → sort the table by `Half-Life (mo)`. Hadoop and Hive are at the bottom.

**To see role convergence:** go to `Role DNA` tab → look at the heatmap. ML Engineer and Platform Engineer show the highest skill overlap.

**To see salary transparency eroding:** go to `Salary Signals` → the transparency chart shows a steady decline over the 18-month window.

---

### Stopping the app

```bash
Ctrl + C    # stops the server
deactivate  # exits the virtual environment
```

---

## Dashboard Tabs

| Tab | What it reveals |
|---|---|
| **Skill Trends** | Demand lines for any skill combo + momentum bar + half-life chart |
| **Skill Intelligence** | Co-occurrence heatmap + full metrics table with velocity and half-life |
| **Role DNA** | PCA scatter of jobs by skill fingerprint + convergence heatmap + cluster explorer |
| **Salary Signals** | Role salary medians + skill premiums + transparency trend + tone vs. pay |
| **Postings Explorer** | Filterable, sortable full posting feed |

---

## The Analytical Ideas Worth Explaining in an Interview

**Skill Half-Life**
Most analytics show current demand. Half-life shows *trajectory*. A skill at 40% demand but falling 2%/month has a 10-month half-life — you have 10 months before it becomes a liability on your resume. That's a fundamentally different insight.

**Role DNA Clustering**
Instead of grouping by job title (the label), group by skill-vector cosine similarity (the content). KMeans on a binary skill presence matrix + PCA for 2D projection. When a "Data Engineer" cluster and an "ML Engineer" cluster overlap in skill-space, the market has converged before the industry has acknowledged it.

**Convergence Score**
Cosine similarity between the average skill vectors of two role titles. A score of 0.85 between Data Engineer and Analytics Engineer means 85% skill overlap — those roles are nearly the same job despite different titles and different salary expectations.

**Salary Transparency Index**
Tracks the monthly % of postings that disclose a salary range. The synthetic data shows it declining — a deliberate simulation of a real market trend. VADER sentiment on job description text then cross-checks whether opaque-comp postings write differently (they do: more "competitive", more "exceptional").

---

## Resume Talking Points

- Built a tech talent intelligence platform that clusters 5,000 job postings by actual skill fingerprint (KMeans + PCA) rather than job title labels, revealing role convergence and boundary erosion across 8 role categories
- Engineered a Skill Half-Life metric computing the months-to-demand-halving for each technology, enabling proactive identification of declining skills before they become resume liabilities
- Implemented multi-dimensional salary signal analysis including skill premium extraction, transparency rate trending, and VADER sentiment scoring on job description language vs. compensation tier
- Designed a role convergence score using cosine similarity on skill-presence vectors to quantify when job titles that carry different salary expectations are becoming the same underlying role

---

## Possible Extensions

| Extension | Impact |
|---|---|
| Live data via PRAW | Stream r/cscareerquestions, r/datascience for real signals |
| LinkedIn scraper | Real job postings replacing synthetic data |
| Salary NLP extractor | Extract salary numbers from free-text using regex + spaCy |
| Time-to-hire signal | Model job posting duration as proxy for demand urgency |
| Resume gap analyzer | Input your skills; get a gap report vs. target role requirements |
| Alert system | Email when a tracked skill crosses a demand threshold |
#   s k i l l r a d a r  
 