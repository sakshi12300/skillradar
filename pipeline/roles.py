"""
Role DNA clustering: groups job titles by actual skill fingerprint,
not by their label. Uses TF-IDF on skill sets + KMeans.

Reveals when two roles are converging (e.g. Data Engineer ≈ ML Engineer)
or when a single role title hides two distinct populations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def build_role_skill_matrix(skill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: rows = job_id, columns = skills, values = 0/1 presence.
    Merges back with role_title for labelling.
    """
    skill_df   = skill_df.copy()
    skill_df["present"] = 1
    pivot = (
        skill_df.pivot_table(index="job_id", columns="skill", values="present", fill_value=0)
        .reset_index()
    )
    # Attach role titles
    role_map = skill_df[["job_id","role_title"]].drop_duplicates()
    return pivot.merge(role_map, on="job_id")


def cluster_roles(role_matrix: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    KMeans on skill presence vectors.
    Returns role_matrix with cluster_id added.
    """
    skill_cols = [c for c in role_matrix.columns if c not in ("job_id", "role_title")]
    X = normalize(role_matrix[skill_cols].values.astype(float))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    role_matrix = role_matrix.copy()
    role_matrix["cluster_id"] = km.fit_predict(X)
    return role_matrix


def pca_embed(role_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    2-D PCA projection for scatter-plot visualisation of role clusters.
    """
    skill_cols = [c for c in role_matrix.columns if c not in ("job_id","role_title","cluster_id")]
    X = normalize(role_matrix[skill_cols].values.astype(float))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    out = role_matrix[["job_id","role_title","cluster_id"]].copy()
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    return out


def top_skills_per_cluster(role_matrix: pd.DataFrame) -> dict[int, list[str]]:
    """Returns the 8 most discriminating skills for each cluster."""
    skill_cols = [c for c in role_matrix.columns if c not in ("job_id","role_title","cluster_id")]
    out = {}
    for cid in sorted(role_matrix["cluster_id"].unique()):
        mask = role_matrix["cluster_id"] == cid
        means = role_matrix.loc[mask, skill_cols].mean().sort_values(ascending=False)
        out[cid] = means.head(8).index.tolist()
    return out


def role_convergence_score(role_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair of official role titles, computes their skill-vector cosine similarity.
    High score = roles are converging in actual skill requirements.
    """
    skill_cols = [c for c in role_matrix.columns if c not in ("job_id","role_title","cluster_id")]
    role_vecs = (
        role_matrix.groupby("role_title")[skill_cols]
        .mean()
    )
    normed = normalize(role_vecs.values)
    sim_matrix = normed @ normed.T
    titles = role_vecs.index.tolist()

    rows = []
    for i, t1 in enumerate(titles):
        for j, t2 in enumerate(titles):
            if i < j:
                rows.append({"role_a": t1, "role_b": t2, "similarity": round(sim_matrix[i, j], 4)})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)
