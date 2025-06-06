# Geographic
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
ev_df = pd.read_csv("C:\Users\navee\OneDrive\Documents\EV Market Segmentation\EV_Dataset.csv")
charging_df = pd.read_csv("ev-charging-stations-india.csv")

# Group by state
ev_state = ev_df.groupby("State")["Total_EVs"].sum().reset_index()
charging_state = charging_df.groupby("State/UT")["Station_Name"].count().reset_index()
charging_state.columns = ['State', 'Charging_Stations']

# Merge both
geo_df = pd.merge(ev_state, charging_state, on="State", how="inner")

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=geo_df.sort_values("Total_EVs", ascending=False), x="State", y="Total_EVs")
plt.xticks(rotation=90)
plt.title("EV Count by State")
plt.show()
 

# Demographic
# Example: Add dummy demographic categories
geo_df["Category"] = geo_df["State"].apply(lambda x: "Metro" if x in ['Delhi', 'Maharashtra', 'Karnataka'] else "Tier-2")

# Compare EV adoption by category
sns.boxplot(data=geo_df, x="Category", y="Total_EVs")
plt.title("EV Adoption by Demographic Category")
plt.show()


# Psychographic
# Tag psychographic labels
def label_lifestyle(state):
    if state in ["Karnataka", "Telangana"]: return "Tech-savvy"
    if state in ["Kerala", "Delhi"]: return "Eco-conscious"
    return "Neutral"

geo_df["Psychographic"] = geo_df["State"].apply(label_lifestyle)

# Visualize
sns.boxplot(data=geo_df, x="Psychographic", y="EVs_per_Station")
plt.title("Psychographic Segmentation: EV Efficiency")
plt.show()


# Behavioral
# Tag behavioral assumptions
def usage_behavior(state):
    if state in ["Delhi", "Maharashtra"]: return "Commuting"
    if state in ["Gujarat", "Haryana"]: return "Logistics"
    return "Mixed"

geo_df["Behavior"] = geo_df["State"].apply(usage_behavior)

# Compare EV infrastructure efficiency
sns.boxplot(data=geo_df, x="Behavior", y="EVs_per_Station")
plt.title("Behavioral Segmentation: EVs per Charging Station")
plt.show()


# Combined Kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = ['Total_EVs', 'Charging_Stations', 'EVs_per_Station']
X = geo_df[features]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
geo_df['Cluster'] = kmeans.fit_predict(X_scaled)


# choose features for clustering
# Normalise the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Total_EVs', 'Charging_Stations', 'EVs_per_Station']])

# Apply Kmeans clustering
from sklearn.cluster import KMeans

# Try different values of k (e.g., 3 to 6)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
import matplotlib.pyplot as plt

plt.scatter(df['Total_EVs'], df['Charging_Stations'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Total EVs')
plt.ylabel('Charging Stations')
plt.title('EV Segmentation by Region')
plt.show()

# or PCA for 2D visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10')
plt.title('Clusters in 2D (PCA)')
plt.show()

#Interpret the clusters
df.groupby('Cluster').mean()

# then interpret the segments
# steps for clustering


# Clustering Script for EV
# Required Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Your Merged/Cleaned Dataset
# Replace with your merged DataFrame or path
df = pd.read_csv("merged_ev_data.csv")  # You can manually create this merged file

# Step 2: Feature Engineering
# Calculate EVs per Charging Station
df['EVs_per_Station'] = df['Total_EVs'] / df['Charging_Stations']
df = df.dropna()

# Step 3: Select Features for Clustering
features = ['Total_EVs', 'Charging_Stations', 'EVs_per_Station']
X = df[features]

# Step 4: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Visualize the Clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set2', s=100)
plt.title('EV Market Segmentation Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Step 7: View Cluster Insights
cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster Summary:\n", cluster_summary)

# Optional: Save clustered data
df.to_csv("ev_segmented_clusters.csv", index=False)


# Seg algorith workflow
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Define node positions
positions = {
    "Data Collection": (0.5, 0.9),
    "Preprocessing": (0.5, 0.8),
    "PCA": (0.3, 0.7),
    "K-Means": (0.15, 0.55),
    "Hierarchical": (0.3, 0.55),
    "DBSCAN": (0.45, 0.55),
    "NLP": (0.65, 0.7),
    "Logistic Regression": (0.6, 0.55),
    "Decision Trees": (0.75, 0.55),
    "Segment Insights": (0.5, 0.4)
}

# Draw nodes
for label, (x, y) in positions.items():
    ax.text(x, y, label, ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle="round,pad=0.4", edgecolor='black', facecolor='lightgrey'))

# Define arrows between nodes
arrows = [
    ("Data Collection", "Preprocessing"),
    ("Preprocessing", "PCA"),
    ("Preprocessing", "NLP"),
    ("PCA", "K-Means"),
    ("PCA", "Hierarchical"),
    ("PCA", "DBSCAN"),
    ("NLP", "Logistic Regression"),
    ("NLP", "Decision Trees"),
    ("K-Means", "Segment Insights"),
    ("Hierarchical", "Segment Insights"),
    ("DBSCAN", "Segment Insights"),
    ("Logistic Regression", "Segment Insights"),
    ("Decision Trees", "Segment Insights")
]

# Draw arrows
for start, end in arrows:
    x_start, y_start = positions[start]
    x_end, y_end = positions[end]
    ax.annotate("",
                xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=1.5))

plt.title("Workflow of Algorithms Used for EV Market Segmentation", fontsize=14)
plt.tight_layout()
plt.show()

