import pandas as pd
import numpy as np
from random_forest import RandomForest
import pickle
import logging
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from joblib import Parallel, delayed
import multiprocessing
import time
from tqdm import tqdm
import warnings

# Filter sklearn warnings about n_quantiles
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model')

# Get number of CPU cores
n_cores = multiprocessing.cpu_count()
logger.info(f"Using {n_cores} CPU cores for parallel processing")

def encode_labels(labels):
    """Convert string labels to numeric values"""
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_to_idx[label] for label in labels]), label_to_idx

def transform_features_parallel(X, features, n_jobs, pbar=None):
    """Transform features using parallel processing"""
    # Group features by transformation type
    quantile_features = ['speechiness', 'instrumentalness', 'liveness']
    standard_features = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']
    cyclical_features = ['key']
    
    # Initialize transformers
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X)))
    ss = StandardScaler()
    
    # Transform features
    X_transformed = X.copy()
    
    # Apply quantile transformation
    for feature in quantile_features:
        if feature in features:
            idx = features.index(feature)
            X_transformed[:, idx] = qt.fit_transform(X[:, [idx]]).ravel()
            if pbar: pbar.update(1)
    
    # Apply standard scaling
    standard_indices = [features.index(f) for f in standard_features if f in features]
    if standard_indices:
        X_transformed[:, standard_indices] = ss.fit_transform(X[:, standard_indices])
        if pbar: pbar.update(1)
    
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
        if pbar: pbar.update(1)
    
    return X_transformed

def train_models():
    try:
        # Start timing
        start_time = time.time()
        logger.info("Starting model training...")
        
        # Step 1: Load Data
        with tqdm(total=4, desc="Initialization") as pbar:
            logger.info("Loading training data...")
            train_df = pd.read_csv('training_data.csv')
            total_samples = len(train_df)
            logger.info(f"Loaded {total_samples:,} training samples")
            pbar.update(1)
            
            # Step 2: Sample Data
            train_df = train_df.sample(frac=0.25, random_state=42)
            used_samples = len(train_df)
            logger.info(f"Using {used_samples:,} samples ({(used_samples/total_samples)*100:.1f}% of data)")
            pbar.update(1)
            
            # Step 3: Feature Selection
            features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'acousticness', 'instrumentalness', 
                       'liveness', 'valence', 'tempo']
            logger.info(f"Selected {len(features)} features")
            pbar.update(1)
            
            # Step 4: Feature Transformation
            logger.info("Transforming features...")
            X = train_df[features].values
            
            # Initialize transformers
            transformers = {
                'quantile': QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X))),
                'standard': StandardScaler()
            }
            
            # Group features by transformation type
            feature_groups = {
                'quantile': ['speechiness', 'instrumentalness', 'liveness'],
                'standard': ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo'],
                'cyclical': ['key'],
                'binary': ['mode']
            }
            
            # Transform features
            X_transformed = X.copy()
            
            # Apply quantile transformation
            quantile_indices = [features.index(f) for f in feature_groups['quantile'] if f in features]
            if quantile_indices:
                X_transformed[:, quantile_indices] = transformers['quantile'].fit_transform(X[:, quantile_indices])
            
            # Apply standard scaling
            standard_indices = [features.index(f) for f in feature_groups['standard'] if f in features]
            if standard_indices:
                X_transformed[:, standard_indices] = transformers['standard'].fit_transform(X[:, standard_indices])
            
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
            
            logger.info(f"Feature matrix shape: {X_transformed.shape}")
            pbar.update(1)
        
        # Step 5: Train Mood Classifier
        logger.info("Training mood classifier...")
        y_mood, mood_label_map = encode_labels(train_df['mood'])
        logger.info(f"Mood labels: {mood_label_map}")
        
        with tqdm(total=50, desc="Training Mood Classifier") as pbar:
            def update_progress(*args):
                pbar.update(1)
            mood_classifier = RandomForest(
                n_trees=50,
                max_depth=3,
                min_samples_split=20,
                n_features=int(np.sqrt(X_transformed.shape[1])),
                n_jobs=n_cores
            )
            mood_classifier.set_progress_callback(update_progress)
            mood_classifier.fit(X_transformed, y_mood)
            mood_classifier.set_progress_callback(None)
        
        # Step 6: Train Activity Classifier
        logger.info("Training activity classifier...")
        y_activity, activity_label_map = encode_labels(train_df['activity'])
        logger.info(f"Activity labels: {activity_label_map}")
        
        with tqdm(total=50, desc="Training Activity Classifier") as pbar:
            def update_progress(*args):
                pbar.update(1)
            activity_classifier = RandomForest(
                n_trees=50,
                max_depth=3,
                min_samples_split=20,
                n_features=int(np.sqrt(X_transformed.shape[1])),
                n_jobs=n_cores
            )
            activity_classifier.set_progress_callback(update_progress)
            activity_classifier.fit(X_transformed, y_activity)
            activity_classifier.set_progress_callback(None)
        
        # Step 7: Save Models and Transformers
        with tqdm(total=3, desc="Saving Models") as pbar:
            logger.info("Saving models and transformers...")
            
            # Calculate feature importance
            original_features = features.copy()
            if 'key' in features:
                key_idx = features.index('key')
                original_features = features[:key_idx] + ['key_sin', 'key_cos'] + features[key_idx+1:]
            
            # Calculate and save mood classifier
            mood_importance = np.zeros(len(original_features))
            for tree, _ in mood_classifier.trees:
                importance = calculate_feature_importance(tree, len(original_features))
                mood_importance += importance
            mood_importance /= len(mood_classifier.trees)
            
            with open('mood_classifier.pkl', 'wb') as f:
                pickle.dump((mood_classifier, mood_label_map, mood_importance), f)
            pbar.update(1)
            
            # Calculate and save activity classifier
            activity_importance = np.zeros(len(original_features))
            for tree, _ in activity_classifier.trees:
                importance = calculate_feature_importance(tree, len(original_features))
                activity_importance += importance
            activity_importance /= len(activity_classifier.trees)
            
            with open('activity_classifier.pkl', 'wb') as f:
                pickle.dump((activity_classifier, activity_label_map, activity_importance), f)
            pbar.update(1)
            
            # Save feature transformers and configuration
            transformer_config = {
                'transformers': transformers,
                'feature_groups': feature_groups,
                'features': features,
                'original_features': original_features
            }
            with open('feature_transformers.pkl', 'wb') as f:
                pickle.dump(transformer_config, f)
            pbar.update(1)
            
            # Log feature importance
            logger.info("\nMood Classification Feature Importance:")
            for feature, importance in zip(original_features, mood_importance):
                logger.info(f"{feature}: {importance:.3f}")
                
            logger.info("\nActivity Classification Feature Importance:")
            for feature, importance in zip(original_features, activity_importance):
                logger.info(f"{feature}: {importance:.3f}")
        
        # Training Summary
        total_time = time.time() - start_time
        logger.info("\nTraining Summary:")
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Samples processed: {used_samples:,}")
        logger.info(f"Processing speed: {used_samples/total_time:.1f} samples/second")
        logger.info(f"Trees trained: {100} (50 per classifier)")
        logger.info(f"CPU cores used: {n_cores}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def calculate_feature_importance(tree, n_features):
    """Calculate feature importance for a single tree"""
    importance = np.zeros(n_features)
    
    def traverse(node, samples=1.0):
        if node is None or node.value is not None:
            return
            
        # Add importance based on information gain and number of samples
        if node.feature is not None and node.feature < n_features:
            importance[node.feature] += node.information_gain * samples
            
        # Recursively traverse children
        if node.left:
            traverse(node.left, samples * 0.5)  # Approximate split ratio
        if node.right:
            traverse(node.right, samples * 0.5)
    
    traverse(tree)
    
    # Normalize importance scores
    if np.sum(importance) > 0:
        importance = importance / np.sum(importance)
    
    return importance

if __name__ == "__main__":
    train_models() 