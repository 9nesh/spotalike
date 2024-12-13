import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# Define the features used during training
FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

def load_data_and_models():
    # Load models and label maps
    mood_classifier, mood_label_map = joblib.load("mood_classifier.pkl")
    activity_classifier, activity_label_map = joblib.load("activity_classifier.pkl")

    # Load validation data
    val_df = pd.read_csv("validation_data.csv")

    # Convert labels to numeric using the maps
    y_mood_val = [mood_label_map[m] for m in val_df['mood_label']]
    y_activity_val = [activity_label_map[a] for a in val_df['activity_label']]

    X_val = val_df[FEATURES].values

    return mood_classifier, activity_classifier, mood_label_map, activity_label_map, X_val, y_mood_val, y_activity_val


def plot_confusion_matrix(clf, X_val, y_true, label_map, title):
    # Predict
    y_pred = clf.predict(X_val)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"{title} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # Invert label_map: {label_str -> int} to {int -> label_str}
    inv_map = {v: k for k, v in label_map.items()}

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = [inv_map[i] for i in range(len(inv_map))]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importances(clf, title):
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(8,5))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        print(f"{title} model does not have feature_importances_ attribute.")


if __name__ == "__main__":
    # Load everything
    (mood_classifier, activity_classifier, 
     mood_label_map, activity_label_map, 
     X_val, y_mood_val, y_activity_val) = load_data_and_models()

    # Plot confusion matrices
    plot_confusion_matrix(mood_classifier, X_val, y_mood_val, mood_label_map, "Mood Classifier Confusion Matrix")
    plot_confusion_matrix(activity_classifier, X_val, y_activity_val, activity_label_map, "Activity Classifier Confusion Matrix")

    # Plot feature importances
    plot_feature_importances(mood_classifier, "Mood Classifier Feature Importances")
    plot_feature_importances(activity_classifier, "Activity Classifier Feature Importances")
