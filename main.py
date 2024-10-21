import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
# This function loads the main dataset and the dataset by genre from CSV files.
@st.cache_data
def load_data():
    main_data = pd.read_csv('data/data.csv')
    data_by_genre = pd.read_csv('data/data_w_genres.csv')
    return main_data, data_by_genre

# Merge datasets and preprocess
# This function merges the main dataset with the genre dataset, preprocesses the data by selecting features,
# one-hot encoding the genre feature, standardizing the feature values, and applying TruncatedSVD for dimensionality reduction.
@st.cache_data
def preprocess_and_reduce(main_data, data_by_genre):
    # Ensure the genre column exists in the main dataset
    if 'genre' not in main_data.columns:
        main_data['genre'] = None  # Initialize with None if missing

    # Merge datasets on the genre column
    merged_data = pd.merge(main_data, data_by_genre, left_on='genre', right_on='genres', how='left')

    # Fill missing genre information in the main dataset
    main_data['genre'] = main_data['genre'].fillna(merged_data['genres'])

    # Select features
    features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    
    # One-hot encode the genre feature
    genre_encoded = pd.get_dummies(main_data['genre'], prefix='genre')
    
    # Combine the features with the encoded genre
    data_features = pd.concat([main_data[features], genre_encoded], axis=1)
    
    # Standardize the feature values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)
    
    # Apply TruncatedSVD (a faster approximation of SVD)
    n_components = min(50, data_scaled.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components)
    data_reduced = svd.fit_transform(data_scaled)
    
    return data_reduced, main_data, svd

# Recommend songs using the reduced matrix
# This function recommends songs based on the input song name by finding matching songs in the dataset
# and returning a list of song options and their indices.
def recommend_songs(input_song, data, max_suggestions=5):
    try:
        # Find indices of songs that match the input
        matching_indices = data[data['name'].str.contains(input_song, case=False, na=False)].index
        
        # Create a list of song options for the dropdown
        song_options = [f"{data.loc[i, 'name']} - {data.loc[i, 'artists']} ({data.loc[i, 'year']})" for i in matching_indices[:max_suggestions]]
        
        return song_options, matching_indices[:max_suggestions]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return [], []

# New function to get recommendations with similarity weighting
# This function gets song recommendations based on the selected song index, reduced data matrix, raw data,
# and an optional similarity feature. It calculates similarities and returns the top recommendations.
def get_recommendations(selected_index, data_reduced, data_raw, similarity_feature=None, num_recommendations=10):
    input_song_vector = data_reduced[selected_index]
    
    if similarity_feature:
        # Get the feature value for the selected song
        feature_value = data_raw.loc[selected_index, similarity_feature]
        
        # Calculate feature differences
        feature_differences = np.abs(data_raw[similarity_feature] - feature_value)
        
        # Normalize feature differences
        feature_differences = (feature_differences - feature_differences.min()) / (feature_differences.max() - feature_differences.min())
        
        # Calculate similarities with extra weight on the selected feature
        similarities = np.dot(data_reduced, input_song_vector) * (1 - feature_differences)
    else:
        similarities = np.dot(data_reduced, input_song_vector)
    
    similar_indices = np.argsort(-similarities)[1:num_recommendations+1]  # Get top recommendations by similarity
    recommendations = data_raw.iloc[similar_indices]
    
    return recommendations

# Calculate similarity scores
# This function calculates similarity scores between the selected song and recommended songs based on selected features.
def calculate_similarity_scores(selected_song, recommendations, features):
    # Include genre features in similarity calculation
    genre_features = [col for col in recommendations.columns if col.startswith('genre_')]
    all_features = features + genre_features
    
    similarity_scores = pd.DataFrame(index=recommendations.index, columns=[f'{f}_similarity' for f in all_features])
    for feature in all_features:
        feature_diff = abs(recommendations[feature] - selected_song[feature])
        feature_max = max(recommendations[feature].max(), selected_song[feature])
        feature_min = min(recommendations[feature].min(), selected_song[feature])
        similarity_scores[f'{feature}_similarity'] = 1 - (feature_diff / (feature_max - feature_min))
    return similarity_scores

# Streamlit app definition
# This function defines the Streamlit app, loads and preprocesses the data, gets user input for song recommendation,
# and displays the recommended songs along with their similarity scores.
def main():
    st.title("Music Recommendation System (SVD-based)")
    st.write("This app recommends songs based on your input using a fast matrix factorization approach (SVD).")
    
    # Load and preprocess the data
    st.write("Loading and preprocessing data...")
    main_data, data_by_genre = load_data()
    data_reduced, data_raw, svd = preprocess_and_reduce(main_data, data_by_genre)
    
    st.write(f"Dataset loaded: {len(data_raw)} songs")
    
    # Get user input for song recommendation
    st.write("### Find Similar Songs")
    song_input = st.text_input("Enter a song name:")
    suggestion_container = st.empty()
    selected_song = st.empty()
    
    # Add "Similar by" feature selection
    similarity_features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    similarity_feature = st.selectbox("Similar by:", ['Overall'] + similarity_features)
    
    if song_input:
        song_options, matching_indices = recommend_songs(song_input, data_raw)
        
        if song_options:
            selected = suggestion_container.radio("Suggestions:", song_options, key="suggestions")
            if selected:
                selected_song.write(f"Selected: {selected}")
                if st.button("Get Recommendations"):
                    # Get the index of the selected song
                    selected_index = matching_indices[song_options.index(selected)]
                    
                    # Display features of the selected song
                    st.write("### Selected Song Features")
                    selected_song_features = data_raw.iloc[selected_index]
                    st.write(selected_song_features)
                    
                    # Get recommendations based on the selected song and similarity feature
                    similarity_feature_param = similarity_feature if similarity_feature != 'Overall' else None
                    recommendations = get_recommendations(selected_index, data_reduced, data_raw, similarity_feature_param)
                    
                    # Calculate similarity scores
                    features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
                    genre_features = [col for col in data_raw.columns if col.startswith('genre_')]
                    
                    # Combine features and genre features to weigh genre features more heavily
                    all_features = features + genre_features
                    similarity_scores = calculate_similarity_scores(selected_song_features, recommendations, all_features)
                    
                    # Combine recommendations with similarity scores
                    recommendations_with_scores = pd.concat([recommendations, similarity_scores], axis=1)
                    
                    st.write("### Recommended Songs")
                    # Display all features of recommended songs along with similarity scores
                    st.dataframe(recommendations_with_scores)
                    
                    # Optionally, you can add a download button for the recommendations
                    csv = recommendations_with_scores.to_csv(index=False)
                    st.download_button(
                        label="Download recommendations as CSV",
                        data=csv,
                        file_name="recommendations.csv",
                        mime="text/csv",
                    )
        else:
            suggestion_container.write("No matching songs found. Try another search term.")

if __name__ == "__main__":
    main()
