import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
import logging
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_model')

def transform_features(X, features):
    """Apply appropriate transformations to features based on their distributions"""
    X_transformed = X.copy()
    
    # Group features by transformation type
    quantile_features = ['speechiness', 'instrumentalness', 'liveness']
    standard_features = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']
    
    # Apply quantile transformation to heavily skewed features
    qt = QuantileTransformer(output_distribution='normal')
    for feature in quantile_features:
        if feature in features:
            idx = features.index(feature)
            X_transformed[:, idx] = qt.fit_transform(X[:, [idx]]).ravel()
    
    # Apply standard scaling to normally distributed features
    ss = StandardScaler()
    standard_indices = [features.index(f) for f in standard_features if f in features]
    if standard_indices:
        X_transformed[:, standard_indices] = ss.fit_transform(X[:, standard_indices])
    
    # Transform key to cyclical features
    if 'key' in features:
        key_idx = features.index('key')
        key_values = X[:, key_idx]
        key_sin = np.sin(2 * np.pi * key_values / 12.0).reshape(-1, 1)
        key_cos = np.cos(2 * np.pi * key_values / 12.0).reshape(-1, 1)
        X_transformed = np.hstack([
            X_transformed[:, :key_idx],
            key_sin,
            key_cos,
            X_transformed[:, key_idx+1:]
        ])
        logger.info(f"Added cyclical encoding for key feature")
    
    return X_transformed

def test_models():
    logger.info("Starting model evaluation...")
    
    # Load validation data
    logger.info("Loading validation data...")
    val_df = pd.read_csv('validation_data.csv')
    logger.info(f"Loaded {len(val_df)} validation samples")
    
    # Select features
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
               'speechiness', 'acousticness', 'instrumentalness', 
               'liveness', 'valence', 'tempo']
    
    X = val_df[features].values
    logger.info(f"Original feature matrix shape: {X.shape}")
    
    # Transform features
    logger.info("Transforming features...")
    X_transformed = transform_features(X, features)
    logger.info(f"Transformed feature matrix shape: {X_transformed.shape}")
    
    # Load and test mood classifier
    logger.info("Loading mood classifier...")
    with open('mood_classifier.pkl', 'rb') as f:
        mood_classifier, mood_label_map, mood_importance = pickle.load(f)
    logger.info(f"Mood labels mapping: {mood_label_map}")
    logger.info(f"Mood feature importance: {mood_importance}")
    
    # Create reverse mapping for predictions
    idx_to_mood = {idx: label for label, idx in mood_label_map.items()}
    true_moods = val_df['mood']
    logger.info("Making mood predictions...")
    pred_moods = np.array([idx_to_mood[idx] for idx in mood_classifier.predict(X_transformed)])
    
    logger.info("\nMood Classification Report:")
    logger.info("\n" + classification_report(true_moods, pred_moods))
    
    # Load and test activity classifier
    logger.info("Loading activity classifier...")
    with open('activity_classifier.pkl', 'rb') as f:
        activity_classifier, activity_label_map, activity_importance = pickle.load(f)
    logger.info(f"Activity labels mapping: {activity_label_map}")
    logger.info(f"Activity feature importance: {activity_importance}")
    
    idx_to_activity = {idx: label for label, idx in activity_label_map.items()}
    true_activities = val_df['activity']
    logger.info("Making activity predictions...")
    pred_activities = np.array([idx_to_activity[idx] for idx in activity_classifier.predict(X_transformed)])
    
    logger.info("\nActivity Classification Report:")
    logger.info("\n" + classification_report(true_activities, pred_activities))
    
    logger.info("Model evaluation completed!")

if __name__ == "__main__":
    test_models() 