import streamlit as st
from song_recommendation_system import run_recommendation_system, SongRecommendationModel

st.set_page_config(page_title="Song Recommendation", layout="wide")

run_recommendation_system()

# If you want to load and display song recommendations or predictions:
# Example:
# st.write("Recommended Songs:", song_recommendations)
# st.write("Artist Popularity Prediction:", artist_popularity)

# Add more functionality specific to the SongRecommendation model
# For instance, you could show song recommendations based on user input:
# st.write("Enter song features to get recommendations:")

# Use relevant code from your SongRecommendation model to interact with users.