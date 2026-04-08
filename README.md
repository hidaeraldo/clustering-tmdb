TMDB Movies Clustering Project

Overview

This project runs **unsupervised clustering** on the TMDB 5000 Movies dataset using **K-Means** and **DBSCAN**. The idea is to group movies using a small set of numeric attributes (budget, revenue, popularity, runtime, vote_average, vote_count) and then visualize how those groups separate.

To help judge clustering quality, it also reports:

- SSE / inertia for K-Means
- the Elbow method (to pick a reasonable K)
- silhouette score (how well clusters separate)

DBSCAN is included mainly to capture dense regions and highlight outliers.

Dataset

- File: `tmdb_5000_movies.csv`
- Source: Kaggle TMDB 5000 Movies Dataset
- Columns used: `budget`, `revenue`, `popularity`, `runtime`, `vote_average`, `vote_count`

What the script does

1. Preprocessing
   - Drops missing values for the selected numeric columns
   - Standardizes features (z-score scaling)

2. K-Means
   - Tries K from 2 to 10
   - Prints SSE and silhouette score for each K
   - Visualizes clusters with PCA
   - Saves plots:
     - `kmeans_elbow_sse.png` (elbow / SSE vs K)
     - `kmeans_silhouette_scores.png` (silhouette vs K)
     - `kmeans_clusters_pca.png` (PCA scatter by cluster)

3. DBSCAN
   - Plots a k-distance curve to guide eps selection (`dbscan_k_distance.png`)
   - Runs DBSCAN and reports cluster counts (including noise)
   - Visualizes clusters with PCA (`dbscan_clusters_pca.png`)

4. Quick summaries
   - Prints K-Means per-cluster averages (in the original feature space)
   - Prints DBSCAN label counts
   - Computes silhouette score for K-Means and (when possible) DBSCAN excluding noise points

How to run

If you don’t already have the dependencies installed, create a virtual environment and install them:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

Then run the script in this folder. It will print the evaluation metrics to the console and write the PNG plots next to the script.

Outputs

- Console:
  - SSE + silhouette scores for K=2–10
  - K-Means cluster averages
  - DBSCAN cluster counts
  - silhouette evaluation summary
- PNG files:
  - `kmeans_elbow_sse.png`
  - `kmeans_silhouette_scores.png`
  - `kmeans_clusters_pca.png`
  - `dbscan_k_distance.png`
  - `dbscan_clusters_pca.png`

Notes

In practice, K-Means tends to group the “bulk” of movies into broad tiers (roughly low/mid/high budget or popularity), while DBSCAN is helpful for flagging unusual points that don’t belong to any dense region. A good K is typically chosen by looking at the elbow curve and the silhouette scores together.