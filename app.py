# Install required libraries
!pip install numpy pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("clustering_2.csv")
print("Original dataset shape:", df.shape)

# Step 2: Keep Only Numeric Data
# -------------------------------
df_num = df.select_dtypes(include=[np.number])

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(df_num)

# Step 3: Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 4: Outlier Removal (ONLY Isolation Forest)
# -------------------------------
iso = IsolationForest(contamination=0.03, random_state=42)
mask = iso.fit_predict(X_scaled) != -1

X_clean = X_scaled[mask]
df_clean = df.iloc[mask].reset_index(drop=True)

print("After outlier removal:", df_clean.shape)

# Step 5: Find Optimal K Automatically
# -------------------------------
K_range = range(2, 11)
silhouette_scores = []
inertia = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_clean)
    
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_clean, labels))

# Best k based on silhouette
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k selected: {optimal_k}")

# Plot Elbow + Silhouette
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("k")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# Step 6: Final Clustering
# -------------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_clean)

df_clean["Cluster"] = labels

print("\nCluster counts:\n", df_clean["Cluster"].value_counts())

# Step 7: Metrics
# -------------------------------
sil = silhouette_score(X_clean, labels)
dbi = davies_bouldin_score(X_clean, labels)
ch = calinski_harabasz_score(X_clean, labels)

print(f"\nSilhouette Score: {sil:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")
print(f"Calinski-Harabasz Index: {ch:.4f}")

# Step 8: Generic Cluster Profiles
# -------------------------------
cluster_profile = df_clean.groupby("Cluster").mean()

print("\n--- Cluster Profiles ---")
print(cluster_profile)

# Optional NLP-style interpretation (generic)
print("\n--- Cluster Insights ---")
for i, row in cluster_profile.iterrows():
    print(f"\nCluster {i}:")
    top_features = row.sort_values(ascending=False).head(3).index.tolist()
    low_features = row.sort_values().head(3).index.tolist()
    
    print(f"  High in: {', '.join(top_features)}")
    print(f"  Low in: {', '.join(low_features)}")
# Step 9: PCA Visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="Set2", s=60)
plt.title("Cluster Visualization (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()







