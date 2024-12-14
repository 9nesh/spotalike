import numpy as np
from collections import Counter
import logging
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('random_forest')

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        logger.debug(f"Fitting decision tree with data shape: {X.shape}")
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or n_labels == 1:
            leaf_value = self._most_common_label(y)
            logger.debug(f"Creating leaf node with value {leaf_value} at depth {depth}")
            return self.Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            logger.debug(f"No valid split found, creating leaf with value {leaf_value}")
            return self.Node(value=leaf_value)

        # Create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # If split doesn't actually split the data
        if not np.any(left_idxs) or not np.any(right_idxs):
            leaf_value = self._most_common_label(y)
            logger.debug(f"Split resulted in empty child, creating leaf with value {leaf_value}")
            return self.Node(value=leaf_value)
        
        logger.debug(f"Creating split node at depth {depth} on feature {best_feature} with threshold {best_threshold}")
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) <= 1:  # Skip if only one unique value
                continue
                
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is not None:
            logger.debug(f"Best split found: feature={best_feature}, threshold={best_threshold}, gain={best_gain}")
        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if not np.any(left_idxs) or not np.any(right_idxs):
            return 0

        n = len(y)
        n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0:
            logger.warning("Empty array passed to _most_common_label, returning 0")
            return 0
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        logger.debug(f"Predicting {len(X)} samples")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        logger.info(f"Initializing RandomForest with {n_trees} trees, max_depth={max_depth}")

    def _train_single_tree(self, X, y, seed):
        np.random.seed(seed)
        # Bootstrap sampling
        idxs = np.random.choice(len(X), size=len(X), replace=True)
        sample_X = X[idxs]
        sample_y = y[idxs]

        # Random feature selection
        if self.n_features is None:
            # At least 1 feature
            self.n_features = max(1, int(np.sqrt(X.shape[1])))

        feature_idxs = np.random.choice(X.shape[1], size=self.n_features, replace=False)
        sample_X = sample_X[:, feature_idxs]

        tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(sample_X, sample_y)
        return (tree, feature_idxs)

    def fit(self, X, y):
        logger.info(f"Training RandomForest on data with shape {X.shape}")
        # If n_features not specified, set a default based on sqrt(#features)
        if self.n_features is None:
            self.n_features = max(1, int(np.sqrt(X.shape[1])))
        logger.info(f"Using {self.n_features} features for random feature selection")

        seeds = [i for i in range(self.n_trees)]
        # Parallel training of each tree
        self.trees = Parallel(n_jobs=-1, verbose=10)(
            delayed(self._train_single_tree)(X, y, seed) for seed in seeds
        )
        
        logger.info("RandomForest training completed")

    def predict(self, X):
        logger.info(f"Making predictions for {len(X)} samples")
        # Collect predictions from all trees
        predictions = np.array([
            self._tree_predict(X, tree, feature_idxs) 
            for tree, feature_idxs in self.trees
        ])
        
        # Majority voting
        final_predictions = np.array([
            Counter(predictions[:, i]).most_common(1)[0][0] 
            for i in range(len(X))
        ])
        logger.info("Predictions completed")
        return final_predictions

    def _tree_predict(self, X, tree, feature_idxs):
        return tree.predict(X[:, feature_idxs])
