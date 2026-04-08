"""
TMDB movies clustering (K-Means + DBSCAN).

This script:
- clusters numeric movie features with K-Means (evaluated for K=2..10)
- runs DBSCAN to highlight dense regions / outliers
- saves evaluation + visualization plots as PNGs
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure Matplotlib/fontconfig can write cache files in sandboxed environments.
_project_dir = Path(__file__).resolve().parent
_cache_dir = _project_dir / ".cache"
_mpl_dir = _cache_dir / "matplotlib"
_fc_dir = _cache_dir / "fontconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
_fc_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("FONTCONFIG_PATH", str(_fc_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    # Fall back to default Matplotlib style if unavailable.
    pass


def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    features = df[FEATURE_COLUMNS].dropna().copy()
    return df, features


def save_line_plot(
    x,
    y,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    line_color: str = "#5B2A86",
    marker_color: str = "#1B998B",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        list(x),
        list(y),
        marker="o",
        linewidth=2.2,
        color=line_color,
        markerfacecolor=marker_color,
        markeredgecolor=line_color,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def save_scatter_plot(xy: np.ndarray, labels: np.ndarray, *, title: str, cmap: str, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, alpha=0.55, edgecolors="none")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def main() -> None:
    df, features = load_features("tmdb_5000_movies.csv")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # K-Means evaluation for K=2..10
    k_range = range(2, 11)
    sse: list[float] = []
    silhouettes: list[float] = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        sse.append(float(kmeans.inertia_))
        sil = float(silhouette_score(scaled, labels))
        silhouettes.append(sil)
        print(f"K={k}: SSE={kmeans.inertia_:.2f}, Silhouette={sil:.4f}")

    save_line_plot(
        k_range,
        sse,
        title="Elbow Method: SSE vs K",
        xlabel="Number of Clusters (K)",
        ylabel="SSE (Inertia)",
        out_path="kmeans_elbow_sse.png",
    )

    # Legacy filename kept for compatibility with older writeups/scripts.
    save_line_plot(
        k_range,
        sse,
        title="Elbow Method for K-Means Optimal K",
        xlabel="Number of clusters (K)",
        ylabel="Inertia",
        out_path="elbow_method.png",
        line_color="#0B3954",
        marker_color="#FFB000",
    )

    save_line_plot(
        k_range,
        silhouettes,
        title="Silhouette Score vs K",
        xlabel="Number of Clusters (K)",
        ylabel="Silhouette Score",
        out_path="kmeans_silhouette_scores.png",
        line_color="#2F3C7E",
        marker_color="#F28C28",
    )

    # Apply K-Means with chosen K
    chosen_k = 3
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled)

    # Keep alignment between original df and filtered features
    df = df.loc[features.index].copy()
    features = features.copy()
    df["kmeans_cluster"] = kmeans_labels
    features["kmeans_cluster"] = kmeans_labels

    # DBSCAN: k-distance curve to guide eps
    neighbors = NearestNeighbors(n_neighbors=10)
    distances, _ = neighbors.fit(scaled).kneighbors(scaled)
    k_distances = np.sort(distances[:, 9], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances, color="#2A9D8F", linewidth=2.0)
    plt.xlabel("Sorted Points")
    plt.ylabel("10th Nearest Neighbor Distance")
    plt.title("DBSCAN k-distance Graph")
    plt.tight_layout()
    plt.savefig("dbscan_k_distance.png")
    plt.show()

    # Pick eps based on the k-distance plot (kept as a fixed example value).
    dbscan_eps = 2.0
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=10)
    dbscan_labels = dbscan.fit_predict(scaled)

    df["dbscan_cluster"] = dbscan_labels
    features["dbscan_cluster"] = dbscan_labels

    print("\nK-Means Cluster Averages:")
    print(features.groupby("kmeans_cluster")[FEATURE_COLUMNS].mean())

    print("\nDBSCAN Cluster Counts:")
    print(pd.Series(dbscan_labels).value_counts())

    kmeans_sil = float(silhouette_score(scaled, kmeans_labels))
    print(f"\nK-Means Silhouette Score: {kmeans_sil:.4f}")

    # DBSCAN silhouette: ignore noise points (-1) and require at least 2 clusters.
    mask = dbscan_labels != -1
    unique_clusters = np.unique(dbscan_labels[mask])
    if unique_clusters.size >= 2:
        dbscan_sil = float(silhouette_score(scaled[mask], dbscan_labels[mask]))
        print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_sil:.4f}")
    else:
        print("DBSCAN Silhouette Score cannot be computed (needs >= 2 clusters excluding noise).")

    # PCA visualizations
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(scaled)

    save_scatter_plot(
        xy,
        kmeans_labels,
        title="K-Means Clusters Visualization (PCA)",
        cmap="Spectral",
        out_path="kmeans_clusters_pca.png",
    )

    # Legacy filename kept for compatibility with older writeups/scripts.
    save_scatter_plot(
        xy,
        kmeans_labels,
        title="K-Means Clusters Visualization (PCA)",
        cmap="cubehelix",
        out_path="clusters_pca.png",
    )

    save_scatter_plot(
        xy,
        dbscan_labels,
        title="DBSCAN Clusters Visualization (PCA)",
        cmap="viridis",
        out_path="dbscan_clusters_pca.png",
    )


if __name__ == "__main__":
    main()