import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pprint import pprint

# Define features (same as in training)
FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

def load_data_and_models():
    # Load models and label maps
    mood_classifier, mood_label_map = joblib.load("mood_classifier.pkl")
    # If needed, also load the activity classifier:
    # activity_classifier, activity_label_map = joblib.load("activity_classifier.pkl")

    # Load validation data
    val_df = pd.read_csv("validation_data.csv")

    y_mood_val = [mood_label_map[m] for m in val_df['mood_label']]
    X_val = val_df[FEATURES].values

    return mood_classifier, mood_label_map, X_val, y_mood_val

def analyze_confusions(cm, label_map):
    # cm is a confusion matrix with rows as true labels and columns as predicted labels
    # label_map is {label_str -> int}, invert it to get {int -> label_str}
    inv_map = {v: k for k, v in label_map.items()}
    labels = [inv_map[i] for i in range(len(inv_map))]

    # Print overall confusion details
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print("   " + "  ".join(labels))
    for i, row in enumerate(cm):
        row_str = [f"{val:3d}" for val in row]
        print(f"{labels[i]:<15}: {'  '.join(row_str)}")

    # Calculate per-label error rates
    true_sums = cm.sum(axis=1)  # sum of each row (true label count)
    misclass_ratios = []
    for i, row in enumerate(cm):
        correct = row[i]
        total = true_sums[i]
        misclassified = total - correct
        misclass_ratio = misclassified / total if total > 0 else 0
        misclass_ratios.append((labels[i], misclass_ratio))

    # Sort by highest misclassification ratio
    misclass_ratios.sort(key=lambda x: x[1], reverse=True)
    print("\nMisclassification ratio per true label (sorted):")
    for label, ratio in misclass_ratios:
        print(f"{label}: {ratio:.2f}")

    # Identify top confusing pairs: which off-diagonal cells are highest relative to the true label count?
    # We want to find where cm[i, j] is large compared to total tracks of true label i.
    confusion_pairs = []
    for i, row in enumerate(cm):
        total = true_sums[i]
        for j, val in enumerate(row):
            if i != j and total > 0:
                # proportion of label i misclassified as j
                proportion = val / total
                confusion_pairs.append((labels[i], labels[j], proportion, val))

    # Sort by proportion of misclassification
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop confusion pairs (label_true -> label_pred, proportion_of_true_misclassified, count):")
    for true_label, pred_label, prop, count in confusion_pairs[:10]:
        print(f"{true_label} -> {pred_label}: {prop:.2%} ({count} tracks)")

if __name__ == "__main__":
    mood_classifier, mood_label_map, X_val, y_mood_val = load_data_and_models()

    # Predict and get confusion matrix
    mood_pred = mood_classifier.predict(X_val)
    cm = confusion_matrix(y_mood_val, mood_pred)

    # Analyze the confusion matrix
    analyze_confusions(cm, mood_label_map)
