# Indian E-Commerce Customer Clusterization 🛒📊

## Introduction

This project applies unsupervised machine learning techniques to the [Indian E-Commerce Customer Behavior & Purchase](https://www.kaggle.com/datasets/kundanbedmutha/indian-e-commerce-customer-behavior-and-purchase) dataset. The goal is to segment customers into meaningful groups based on their browsing and purchasing behavior, enabling targeted marketing strategies.

The notebook demonstrates a complete clustering workflow: exploratory data analysis, preprocessing, feature engineering, dimensionality reduction, multi-algorithm clustering, and business-oriented interpretation of results. The main goal is to discover hidden patterns in customer behavior such as purchasing habits, session activity, revenue generation, and engagement levels.

---

## Methodology

### Exploratory Data Analysis (EDA)
- Visualized feature distributions using histograms, violin plots, and box plots
- Analyzed correlations between attributes via a heatmap
- Identified skewed features and potential outliers
- Examined unique value ranges across all 29 columns

### Data Preprocessing
- Removed non-informative columns (`customer_id`, `session_id`, `visit_date`, `product_id`)
- Ordinal encoding of `session_duration_bucket` (Very Short → Very Long mapped to 1–4)
- Removed highly correlated features: `revenue_normalized` (correlation 1.0 with `revenue`) and `session_duration_bucket` (correlation 0.97 with `time_on_site_sec`)
- Log transformation applied to skewed columns: `discount_percent`, `discount_amount`, `revenue`, `review_text`, `review_helpful_votes`
- Scaling with **RobustScaler** (robust to outliers, uses IQR)

### Outlier Detection — Consensus Approach
Three methods were combined; a point is flagged as an outlier if **at least 2 out of 3** methods agree:
- Isolation Forest
- Local Outlier Factor (LOF)
- Elliptic Envelope

### Feature Engineering
Since the original dataset contains one row per session, features were aggregated per customer to enable customer-level segmentation:
- `num_sessions`, `num_purchases`, `total_revenue`, `avg_revenue_per_purchase`
- `total_quantity`, `avg_discount_percent`, `total_discount_amount`
- `avg_helpful_votes`, `conversion_rate`, seasonal activity index

### Dimensionality Reduction
- **PCA** retaining 95% of variance, with 2D and 3D visualizations before and after clustering

---

## Clustering Algorithms

All algorithms were applied to PCA-reduced data. The optimal number of clusters (k=4) was determined using the **Elbow Method** and **Silhouette Score**.

- **K-Means** — classic centroid-based clustering
- **DBSCAN** — density-based, with parameters tuned via k-distance graph (eps=2.3, min_samples=15)
- **Agglomerative Clustering** — hierarchical, bottom-up approach
- **Gaussian Mixture Model (GMM)** — soft probabilistic assignment
- **BIRCH** — efficient tree-based clustering, suitable for larger datasets

---

## Customer Segments

All models (except DBSCAN) consistently identified the same customer profiles:

| Cluster | Name | Key Characteristics | Recommended Action |
|---|---|---|---|
| 0 | **Active Buyers / Reviewers** | High conversion rate (~0.55), write helpful reviews, solid revenue | Loyalty program, early access, review incentives |
| 1 | **Window Shoppers / Cold Leads** | Almost zero purchases, fewest sessions, minimal discounts | Win-back email campaign, welcome discount, personalized recommendations |
| 2 | **Discount Hunters** | Many sessions, highest avg. discount (11%), but near-zero conversion | Flash sales, urgency tactics ("2h left"), exit-intent popups |
| 3 | **VIP / High-Value Buyers** | Most sessions, purchases, revenue, and quantity across all seasons | VIP tier, free shipping, bundle offers, dedicated account manager |

---

## Evaluation Metrics

| Metric | Description | Optimum |
|---|---|---|
| **Silhouette Score** | Measures cluster separation and cohesion | Higher is better (max 1) |
| **Davies-Bouldin Index** | Measures compactness and inter-cluster distance | Lower is better |
| **Calinski-Harabasz Score** | Ratio of between-cluster to within-cluster variance | Higher is better |

K-Means and GMM achieved the best metric values overall. DBSCAN produced only 2 meaningful clusters on this dataset, suggesting it is not the optimal choice for this type of customer data.

---

## Results

- K-Means, Agglomerative, GMM, and BIRCH all converged on the **same 4-cluster structure**, confirming the robustness of the segmentation.
- The consistent identification of a "Discount Hunter" segment (high session count, high discount, zero conversion) provides a direct, actionable insight for the marketing team.
- PCA clustering outperformed raw feature clustering based on silhouette and inertia values.
- DBSCAN was less suitable due to the continuous, non-spherical nature of the data.

---

## Tech Stack

- **Python** — pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **Clustering** — KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, BIRCH
- **Preprocessing** — RobustScaler, IsolationForest, LOF, EllipticEnvelope
- **Dimensionality Reduction** — PCA
