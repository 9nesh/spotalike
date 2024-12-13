import streamlit as st

# Page config - this should be the first Streamlit command
st.set_page_config(
    page_title="SpotaLike - Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Enhanced Custom CSS
st.markdown("""
<style>
/* Dark theme colors */
:root {
    --spotify-black: #121212;
    --spotify-dark-gray: #181818;
    --spotify-light-gray: #282828;
    --spotify-green: #1DB954;
    --spotify-white: #FFFFFF;
    --spotify-text-gray: #B3B3B3;
}

/* Base theme overrides */
.stApp {
    background-color: var(--spotify-black);
}

.stMarkdown {
    color: var(--spotify-white) !important;
}

/* Typography improvements */
h1, h2, h3 {
    color: var(--spotify-white) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* Enhanced welcome card */
.welcome-card {
    background: linear-gradient(145deg, var(--spotify-dark-gray), var(--spotify-light-gray));
    padding: 40px;
    border-radius: 16px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}

/* Improved feature cards */
.feature-highlight {
    background: linear-gradient(145deg, var(--spotify-dark-gray), var(--spotify-light-gray));
    padding: 25px;
    border-radius: 12px;
    margin: 15px 0;
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.feature-highlight:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    border-color: var(--spotify-green);
}

/* Enhanced buttons */
.spotify-button {
    background: linear-gradient(145deg, var(--spotify-green), #1ed760);
    color: black;
    padding: 14px 28px;
    border-radius: 30px;
    text-decoration: none;
    font-weight: 700;
    display: inline-block;
    margin: 15px 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(29,185,84,0.3);
}

.spotify-button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 16px rgba(29,185,84,0.4);
}

/* Custom progress bar */
.stProgress > div > div > div {
    background-color: var(--spotify-green) !important;
}

/* Custom select boxes */
.stSelectbox label {
    color: var(--spotify-white) !important;
}

.stSelectbox > div > div {
    background-color: var(--spotify-dark-gray) !important;
    border-color: var(--spotify-light-gray) !important;
}
</style>
""", unsafe_allow_html=True)

# Enhanced Welcome Section with Animation
st.markdown("""
<div class="welcome-card">
    <h1>🎵 Welcome to SpotaLike</h1>
    <p style="color: var(--spotify-white); font-size: 1.3em; margin: 20px 0; font-weight: 300;">
        Discover Your Perfect Sound with AI-Powered Music Recommendations
    </p>
    <a href="#get-started" class="spotify-button">Get Started</a>
</div>
""", unsafe_allow_html=True)

# Enhanced Features Overview
st.markdown("## ✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-highlight">
        <h3>🎭 Mood-Based</h3>
        <p style="color: var(--spotify-text-gray); line-height: 1.6;">
            Our AI understands your emotions and suggests the perfect tracks to match your current mood
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-highlight">
        <h3>🏃‍♂️ Activity-Driven</h3>
        <p style="color: var(--spotify-text-gray); line-height: 1.6;">
            Whether you're working out, studying, or relaxing, get tailored playlists for any activity
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-highlight">
        <h3>🎧 Smart Playback</h3>
        <p style="color: var(--spotify-text-gray); line-height: 1.6;">
            Seamless Spotify integration for instant listening and playlist creation
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Quick Start Guide
st.markdown('<div id="get-started"></div>', unsafe_allow_html=True)
st.markdown("## 🚀 Quick Start")
st.markdown("""
<div class="feature-highlight">
    <p style="color: var(--spotify-white); line-height: 2;">
        <span style="color: var(--spotify-green); font-weight: 600;">1.</span> Connect your Spotify account<br>
        <span style="color: var(--spotify-green); font-weight: 600;">2.</span> Tell us your current mood and activity<br>
        <span style="color: var(--spotify-green); font-weight: 600;">3.</span> Adjust your preferences<br>
        <span style="color: var(--spotify-green); font-weight: 600;">4.</span> Get personalized recommendations<br>
        <span style="color: var(--spotify-green); font-weight: 600;">5.</span> Create and save custom playlists
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: var(--spotify-text-gray); margin-bottom: 10px;'>
        Built with ❤️ using Python, Streamlit, and Spotify API
    </p>
    <p style='color: var(--spotify-text-gray); font-size: 0.9em;'>
        © 2024 SpotaLike | <a href="#" style="color: var(--spotify-green); text-decoration: none;">Privacy Policy</a> | 
        <a href="#" style="color: var(--spotify-green); text-decoration: none;">Terms of Service</a>
    </p>
</div>
""", unsafe_allow_html=True) 