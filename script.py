import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline')

def main():
    logger.info("Starting pipeline...")

    # Step 1: Load Data
    logger.info("Loading data.csv...")
    df = pd.read_csv("data.csv")

    # Basic check for required features
    required_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    for feat in required_features:
        if feat not in df.columns:
            logger.error(f"Missing required feature: {feat}")
            return

    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

    # Drop rows with missing values in required features
    df = df.dropna(subset=required_features)
    logger.info(f"After dropping missing values, {len(df)} rows remain")

    # Step 2: Preprocess (Scaling)
    X = df[required_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Data scaling completed.")

    # Step 3: Unsupervised Clustering for Mood
    # Choose k for mood clusters
    mood_k = 5
    logger.info(f"Performing K-Means with k={mood_k} for mood clusters...")
    kmeans_mood = KMeans(n_clusters=mood_k, random_state=42)
    mood_labels = kmeans_mood.fit_predict(X_scaled)
    df['mood_cluster'] = mood_labels
    logger.info("Mood clustering completed.")

    # Inspect clusters
    mood_cluster_summary = df.groupby('mood_cluster')[required_features].mean()
    logger.info("Mood cluster summary (means of each feature):")
    logger.info(mood_cluster_summary)

    # Assign mood labels based on cluster characteristics (example mapping)
    # Adjust after inspecting mood_cluster_summary as needed
    # Example heuristic:
    #   Check cluster summary and decide labels:
    #   Let's say:
    #   0 -> "Happy"
    #   1 -> "Sad"
    #   2 -> "Party"
    #   3 -> "Peaceful/Relaxed"
    #   4 -> "Chill"
    mood_cluster_to_label = {
        0: "Happy",
        1: "Sad",
        2: "Party",
        3: "Peaceful/Relaxed",
        4: "Chill"
    }

    df['mood_label'] = df['mood_cluster'].map(mood_cluster_to_label)

    # Optional: Unsupervised Clustering for Activity
    # For simplicity, reuse same clustering method or do a separate one:
    activity_k = 6
    logger.info(f"Performing K-Means with k={activity_k} for activity clusters...")
    kmeans_activity = KMeans(n_clusters=activity_k, random_state=42)
    activity_labels = kmeans_activity.fit_predict(X_scaled)
    df['activity_cluster'] = activity_labels
    logger.info("Activity clustering completed.")

    activity_cluster_summary = df.groupby('activity_cluster')[required_features].mean()
    logger.info("Activity cluster summary (means of each feature):")
    logger.info(activity_cluster_summary)

    # Assign activity labels (example heuristic):
    #   0 -> "Workout"
    #   1 -> "Study/Focus"
    #   2 -> "Party"
    #   3 -> "Background"
    #   4 -> "Driving"
    #   5 -> "General"
    # Adjust as needed.
    activity_cluster_to_label = {
        0: "Workout",
        1: "Study/Focus",
        2: "Party",
        3: "Background",
        4: "Driving",
        5: "General"
    }

    df['activity_label'] = df['activity_cluster'].map(activity_cluster_to_label)

    # Step 4: Introduce noise in labels (optional)
    noise_fraction = 0.1
    num_noise = int(len(df) * noise_fraction)
    logger.info(f"Introducing label noise in {num_noise} rows...")
    mood_choices = df['mood_label'].unique().tolist()
    activity_choices = df['activity_label'].unique().tolist()

    noise_indices = random.sample(range(len(df)), num_noise)
    for idx in noise_indices:
        # Choose a different mood label randomly
        current_mood = df.iloc[idx]['mood_label']
        mood_options = [m for m in mood_choices if m != current_mood]
        df.iat[idx, df.columns.get_loc('mood_label')] = random.choice(mood_options)

        # Choose a different activity label randomly
        current_activity = df.iloc[idx]['activity_label']
        activity_options = [a for a in activity_choices if a != current_activity]
        df.iat[idx, df.columns.get_loc('activity_label')] = random.choice(activity_options)
    logger.info("Label noise introduced.")

    # Step 5: Split into Training and Validation Sets
    logger.info("Splitting into training and validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("training_data.csv", index=False)
    val_df.to_csv("validation_data.csv", index=False)
    logger.info(f"Training set: {len(train_df)} rows, Validation set: {len(val_df)} rows")

    # Step 6: Encode Labels
    mood_set = train_df['mood_label'].unique()
    mood_label_map = {m: i for i, m in enumerate(mood_set)}

    activity_set = train_df['activity_label'].unique()
    activity_label_map = {a: i for i, a in enumerate(activity_set)}

    y_mood = np.array([mood_label_map[m] for m in train_df['mood_label']])
    y_activity = np.array([activity_label_map[a] for a in train_df['activity_label']])
    X_train = train_df[required_features].values

    # Step 7: Train Supervised Models
    logger.info("Training mood classifier...")
    mood_classifier = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=10, random_state=42)
    mood_classifier.fit(X_train, y_mood)
    logger.info("Mood classifier trained.")

    logger.info("Training activity classifier...")
    activity_classifier = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=10, random_state=42)
    activity_classifier.fit(X_train, y_activity)
    logger.info("Activity classifier trained.")

    # Step 8: Validate Models
    X_val = val_df[required_features].values
    y_mood_val = np.array([mood_label_map[m] for m in val_df['mood_label']])
    y_activity_val = np.array([activity_label_map[a] for a in val_df['activity_label']])

    mood_pred = mood_classifier.predict(X_val)
    activity_pred = activity_classifier.predict(X_val)

    mood_accuracy = accuracy_score(y_mood_val, mood_pred)
    activity_accuracy = accuracy_score(y_activity_val, activity_pred)

    mood_f1 = f1_score(y_mood_val, mood_pred, average='weighted')
    activity_f1 = f1_score(y_activity_val, activity_pred, average='weighted')

    logger.info(f"Mood classifier - Accuracy: {mood_accuracy:.4f}, F1: {mood_f1:.4f}")
    logger.info(f"Activity classifier - Accuracy: {activity_accuracy:.4f}, F1: {activity_f1:.4f}")

    # Step 9: Save Models and Mappings
    logger.info("Saving models and label maps...")
    dump((mood_classifier, mood_label_map), "mood_classifier.pkl")
    dump((activity_classifier, activity_label_map), "activity_classifier.pkl")
    logger.info("Models and mappings saved successfully.")

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    main()
