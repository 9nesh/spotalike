# Music Recommendation System

This project is a music recommendation system built using Streamlit and machine learning techniques. It leverages matrix factorization (SVD) to recommend songs based on user input.

## Features

- **Load and Preprocess Data**: The system loads datasets containing song features and genres, preprocesses them by selecting relevant features, one-hot encoding genres, and standardizing feature values.
- **Dimensionality Reduction**: Uses Truncated SVD to reduce the dimensionality of the feature space for faster computation.
- **Song Recommendation**: Recommends songs based on user input by calculating similarity scores.
- **Similarity Weighting**: Allows users to weigh recommendations based on specific features like danceability, energy, etc.
- **Streamlit Interface**: Provides an interactive web interface for users to input song names and receive recommendations.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Load Data**: The app automatically loads and preprocesses the data upon startup.
2. **Input Song Name**: Enter a song name in the text input field to find similar songs.
3. **Select Similarity Feature**: Choose a feature to weigh more heavily in the recommendations.
4. **Get Recommendations**: Click the "Get Recommendations" button to view similar songs and their features.

## Code Overview

- **Data Loading and Preprocessing**: 
  - `load_data()` function loads the datasets.
  - `preprocess_and_reduce()` function merges datasets, preprocesses features, and applies Truncated SVD.
  - Code reference: 
    ```python:main.py
    startLine: 9
    endLine: 48
    ```

- **Recommendation Functions**:
  - `recommend_songs()` function finds songs matching the input name.
  - `get_recommendations()` function calculates similarity scores and returns top recommendations.
  - Code reference:
    ```python:main.py
    startLine: 53
    endLine: 90
    ```

- **Similarity Calculation**:
  - `calculate_similarity_scores()` function calculates similarity scores between the selected song and recommended songs.
  - Code reference:
    ```python:main.py
    startLine: 94
    endLine: 105
    ```

- **Streamlit App**:
  - The `main()` function defines the Streamlit app, handles user input, and displays recommendations.
  - Code reference:
    ```python:main.py
    startLine: 110
    endLine: 178
    ```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- This project uses the [Streamlit](https://streamlit.io/) library for building the web interface.
- The machine learning components are powered by [scikit-learn](https://scikit-learn.org/).
- Team: Inesh Tickoo, Mostafa Anwari, and Nick Davis 
