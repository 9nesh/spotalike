import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

# Create output directory for visualizations and text files
output_dir = 'viz'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset that already has mood_cluster/activity_cluster columns
df = pd.read_csv("training_data.csv")

# Define the same features used during clustering
FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]

# Extract the feature matrix
X = df[FEATURES].values

# Create numeric mappings for mood and activity labels
mood_map = {label: idx for idx, label in enumerate(sorted(df['mood'].unique()))}
activity_map = {label: idx for idx, label in enumerate(sorted(df['activity'].unique()))}

# Convert labels to numeric values
mood_numeric = np.array([mood_map[m] for m in df['mood'].values])
activity_numeric = np.array([activity_map[a] for a in df['activity'].values])

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Create color maps
mood_colors = plt.cm.viridis(np.linspace(0, 1, len(mood_map)))
activity_colors = plt.cm.plasma(np.linspace(0, 1, len(activity_map)))

# Plot the PCA results colored by mood
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=mood_numeric, cmap='viridis', alpha=0.7)
plt.title("PCA - Songs Colored by Mood")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

# Create legend for moods
handles = [plt.scatter([], [], c=[mood_colors[i]], label=mood) 
          for mood, i in mood_map.items()]
plt.legend(handles=handles, title="Moods", loc='best')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_mood.png'))
plt.close()

# Plot the PCA results colored by activity
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=activity_numeric, cmap='plasma', alpha=0.7)
plt.title("PCA - Songs Colored by Activity")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

# Create legend for activities
handles = [plt.scatter([], [], c=[activity_colors[i]], label=activity)
          for activity, i in activity_map.items()]
plt.legend(handles=handles, title="Activities", loc='best')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_activity.png'))
plt.close()

# Prepare output for explained variance ratio
output_text = []
output_text.append("\nExplained variance ratio by principal components:")
output_text.append(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
output_text.append(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
output_text.append(f"Total: {sum(pca.explained_variance_ratio_):.3f}")

# Print feature contributions
feature_weights = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=FEATURES
)
output_text.append("\nFeature contributions to principal components:")
output_text.append(feature_weights.to_string())

# Save output to a text file
with open(os.path.join(output_dir, 'pca_output.txt'), 'w') as f:
    f.write("\n".join(output_text))
