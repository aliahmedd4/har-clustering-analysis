# Human Activity Recognition — Unsupervised Clustering

**GIU Introduction to Data Science · Spring 2026 · Project 3**

---

## Problem Statement

Given raw smartphone sensor readings (accelerometer + gyroscope), can we **discover distinct human motion patterns without using any pre-existing activity labels**?

This project applies unsupervised clustering to the HAR dataset to identify six natural activity groups, profile their physical characteristics, detect anomalous movement windows, and compare four different clustering algorithms to determine which best captures the "Human Motion Protocol."

---

## Dataset

| Property | Value |
|----------|-------|
| File | `human_activity_master.csv` |
| Samples | 7,352 |
| Features | 561 smartphone sensor features |
| Subjects | 30 individuals |
| Feature types | Time-domain & frequency-domain statistics from accelerometer and gyroscope |
| Missing values | None |

> **Note:** The dataset file is excluded from this repository (listed in `.gitignore`). It is the GIU Introduction to Data Science Spring 2026 course dataset.

---

## Approach

### 1 · Data Preparation
- **StandardScaler** applied to all 561 features — essential for distance-based clustering because different sensor-derived statistics span incompatible scales
- **Correlation-based redundancy removal**: computed the 561 × 561 absolute Pearson correlation matrix and removed one feature from every pair with |r| > 0.95
  - **308 features dropped → 253 features retained**

### 2 · Choosing k
- **Elbow Method**: plotted WCSS (inertia) for k = 2 … 10; curve bends around k = 4–6
- **Silhouette Analysis**: scores computed for k = 2 … 10 (sample_size = 2000 for efficiency)
- **Chosen k = 6** — confirmed by both quantitative plots and domain knowledge (the HAR protocol records exactly 6 distinct activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Lying)

### 3 · Discovery & Evaluation
- **Motion Profiling**: cluster centroids examined across body-acceleration mean features to label clusters as *Dynamic (Active)* vs *Static (Quiet)*
- **PCA Visualisation**: 2-D projection coloured by cluster label; clear separation between walking and stationary activities along PC1
- **Anomaly Detection**: top 5 % of samples by distance to their assigned centroid flagged as anomalies; boundary distribution in PCA space suggests activity-transition windows rather than sensor noise

### 4 · Advanced Comparisons
Four methods compared at k = 6:
| Method | Description |
|--------|-------------|
| K-Means | Hard assignment, minimises WCSS |
| Agglomerative (Ward) | Hierarchical bottom-up, minimises within-cluster variance at each merge |
| GMM | Soft/probabilistic assignment — each sample gets a probability over all components |
| BIRCH | Memory-efficient CF-Tree approach, suited for large datasets |

---

## Key Results

| Metric | Value |
|--------|-------|
| Original features | 561 |
| Features retained after pruning | 253 (dropped 308) |
| Chosen k | **6** |
| K-Means Silhouette Score | **0.0755** (Rank #1) |
| Agglomerative (Ward) Silhouette | 0.0485 (Rank #2) |
| BIRCH Silhouette Score | 0.0485 (Rank #3) |
| GMM Silhouette Score | 0.0420 (Rank #4) |
| PCA variance captured (PC1 + PC2) | **36.97 %** (28.80 % + 8.17 %) |
| Anomalies detected (top 5 %) | **368 samples** (5.01 %) |
| Best clustering method | **K-Means** |

**Conclusion:** K-Means with k = 6 best represents the Human Motion Protocol. The six HAR activity clusters are approximately spherical and compact in sensor feature space — precisely the geometry K-Means optimises. GMM is the recommended complement for probabilistic transition modelling.

---

## Running the Project Locally

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repository
git clone https://github.com/aliahmedd4/har-clustering-analysis.git
cd har-clustering-analysis

# Install dependencies
pip install -r requirements.txt

# Place the dataset in the project root
# (human_activity_master.csv must be present — not included in repo)

# Launch Jupyter
jupyter notebook Project3_HAR_Clustering.ipynb
```

### Running all cells

Open `Project3_HAR_Clustering.ipynb` and select **Kernel → Restart & Run All**.
Total execution time is approximately 3–5 minutes (correlation matrix computation is the longest step).

---

## Repository Structure

```
har-clustering-analysis/
├── Project3_HAR_Clustering.ipynb   # Main notebook (executed, outputs included)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Excludes dataset, checkpoints, caches
```

---

## Contributors

| Name | Role |
|------|------|
| Ali Sherif | Lead analysis, clustering implementation |
| Ahmed Rashad | Visualisation, anomaly detection |
| Asser Ehab | Preprocessing, comparative evaluation |
