import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data (you might want to create a shared module for data loading to avoid duplication)
@st.cache_data
def load_data():
    # Replace with actual dataset path
    data = pd.read_csv('data/data.csv')
    return data

def main():
    st.title("Music Dataset Visualizations")

    data = load_data()

    st.write("## Dataset Overview")
    st.write(f"Total number of songs: {len(data)}")

    # Distribution of genres
    st.write("### Distribution of Genres")
    genre_counts = data['genre'].value_counts()
    fig = px.pie(values=genre_counts.values, names=genre_counts.index, title="Genre Distribution")
    st.plotly_chart(fig)

    # Correlation heatmap
    st.write("### Feature Correlation Heatmap")
    numeric_columns = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    corr = data[numeric_columns].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    st.plotly_chart(fig)

    # Scatter plot of two features
    st.write("### Scatter Plot: Energy vs. Valence")
    fig = px.scatter(data, x='energy', y='valence', color='genre', hover_name='name', title="Energy vs. Valence by Genre")
    st.plotly_chart(fig)

    # Distribution of a numeric feature
    st.write("### Distribution of Danceability")
    fig = px.histogram(data, x='danceability', nbins=30, title="Distribution of Danceability Scores")
    st.plotly_chart(fig)

    # Box plot of a feature across genres
    st.write("### Tempo Across Genres")
    fig = px.box(data, x='genre', y='tempo', title="Tempo Distribution Across Genres")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()