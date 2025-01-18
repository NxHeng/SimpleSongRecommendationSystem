import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

def run_linear_regression_app():
    # Load the trained model
    model = joblib.load("best_xgb_model.pkl")

    # Simulated dataset to refit scalers (replace this with your training dataset)
    training_data = pd.read_csv("integrated_consolidated_data.csv")  # Replace with the actual file

    # Recreate and fit the scalers
    scaler_album_total_tracks = RobustScaler()
    scaler_album_total_tracks.fit(np.log1p(training_data[['Album_Total_Tracks']]))

    scaler_artist_followers = RobustScaler()
    scaler_artist_followers.fit(np.log1p(training_data[['Artist_Followers']]))

    scaler_loudness_tempo = RobustScaler()
    training_data['Loudness'] = np.log1p(training_data['Loudness'] - training_data['Loudness'].min() + 1)
    training_data['Tempo'] = np.log1p(training_data['Tempo'])
    scaler_loudness_tempo.fit(training_data[['Loudness', 'Tempo']])

    scaler_duration = RobustScaler()
    scaler_duration.fit(np.log1p(training_data[['Duration']]))

    # Streamlit app interface
    st.title("Track Popularity Prediction")

    # Input fields for user to enter values
    danceability = st.number_input("Danceability", min_value=0.0, max_value=1.0, value=0.5, format="%.3f")
    energy = st.number_input("Energy", min_value=0.0, max_value=1.0, value=0.7, format="%.3f")
    loudness = st.number_input("Loudness (Negative Values)", min_value=-60.0, max_value=0.0, value=-20.0, format="%.3f")
    speechiness = st.number_input("Speechiness", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
    acousticness = st.number_input("Acousticness", min_value=0.0, max_value=1.0, value=0.1, format="%.4f")
    instrumentalness = st.number_input("Instrumentalness", min_value=0.0, value=0.4, format="%.4f")  # Allow scientific notation
    liveness = st.number_input("Liveness", min_value=0.0, max_value=1.0, value=0.2, format="%.3f")
    valence = st.number_input("Valence", min_value=0.0, max_value=1.0, value=0.5, format="%.3f")
    tempo = st.number_input("Tempo", min_value=0.0, max_value=250.0, value=120.0, format="%.3f")
    album_total_tracks = st.number_input("Album Total Tracks", min_value=1, value=10, step=1)
    album_popularity = st.number_input("Album Popularity", min_value=0, max_value=100, value=60, step=1)
    duration = st.number_input("Duration", min_value=0.0, max_value=999999.0, value=394821.000000, format="%.6f")
    artist_followers = st.number_input("Artist Followers", min_value=0.0, value=6442243.00)
    artist_popularity = st.number_input("Artist Popularity", min_value=0, max_value=100, value=70, step=1)


    # Preprocess user inputs (apply necessary transformations)
    album_total_tracks = np.log1p(album_total_tracks)
    artist_followers = np.log1p(artist_followers)
    loudness = np.log1p(loudness - training_data['Loudness'].min() + 1)
    tempo = np.log1p(tempo)
    duration = np.log1p(duration)

    # Scale inputs
    album_total_tracks_scaled = scaler_album_total_tracks.transform([[album_total_tracks]])[0, 0]
    artist_followers_scaled = scaler_artist_followers.transform([[artist_followers]])[0, 0]
    loudness_tempo_scaled = scaler_loudness_tempo.transform([[loudness, tempo]])[0]
    duration_scaled = scaler_duration.transform([[duration]])[0, 0]

    # Combine all features
    features = np.array([
        album_total_tracks_scaled, acousticness, liveness, loudness_tempo_scaled[0],
        duration_scaled, instrumentalness, loudness_tempo_scaled[1], artist_followers_scaled,
        valence, artist_popularity, speechiness, danceability, energy, album_popularity
    ]).reshape(1, -1)

    # Make prediction on button click
    if st.button("Predict"):
        prediction = model.predict(features)
        st.write(f"Predicted Track Popularity: {prediction[0]:.2f}")
