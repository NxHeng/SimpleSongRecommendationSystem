import streamlit as st
import requests
import pickle
from sklearn.neighbors import NearestNeighbors
import base64
import urllib.parse
import os
from dotenv import load_dotenv
import time

# Load environment variables from the .env file
load_dotenv()

st.set_page_config(layout="wide")

# Access the environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
AUTH_URL = os.getenv("AUTH_URL")
TOKEN_URL = os.getenv("TOKEN_URL")

# Scopes for Spotify Playback
SCOPES = "streaming user-read-playback-state user-modify-playback-state app-remote-control"

# Function to validate the access token
def is_token_valid(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://api.spotify.com/v1/me", headers=headers)
    return response.status_code == 200

# Function to refresh the access token (if you store the refresh token)
def refresh_access_token(refresh_token):
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode("ascii")
    headers = {"Authorization": f"Basic {auth}"}
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        st.error("Error refreshing access token.")
        return None

# Function to get the authorization URL
def get_authorization_url():
    scope = SCOPES  # Adjust scopes as needed
    auth_url = f"{AUTH_URL}?response_type=code&client_id={SPOTIFY_CLIENT_ID}&scope={urllib.parse.quote(scope)}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
    return auth_url

# Function to get the access token from the Spotify API
def get_access_token(auth_code):
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": REDIRECT_URI
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        st.error("Error while getting access token")
        return None
    
def clean_album_id(album_id):
    # Remove the 'spotify:album:' prefix if it exists
    if album_id.startswith("spotify:album:"):
        return album_id.replace("spotify:album:", "")
    return album_id

def get_album_image(album_id, access_token):
    album_id = clean_album_id(album_id)  # Clean the album_id by removing 'spotify:album:'

    if not album_id:  # Check if album_id is empty or None
        st.error("Invalid or missing album ID.")
        return None
    
    # Spotify API URL to get album details
    url = f"https://api.spotify.com/v1/albums/{album_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        album_data = response.json()
        # Get the album image URL (widest image)
        if 'images' in album_data and len(album_data['images']) > 0:
            return album_data['images'][0]['url']
        else:
            st.error("Album image not found.")
            return None
    else:
        st.error(f"Error fetching album image. Status code: {response.status_code}. Response: {response.text}")
        return None

# Function to display song card
def display_song_card(rec, access_token):
    song_url = f"https://open.spotify.com/track/{rec['song_id'].split(':')[2]}"
    album_image_url = get_album_image(rec['album_id'], access_token)

    # Card-like format
    with st.expander(f"**{rec['song_name']}** - {rec['artist_name']}", expanded=True):
        col1, col2 = st.columns([2, 6])  # Two columns: 2 for image, 6 for text

        with col1:
            # Display album image (with use_container_width for responsive image scaling)
            if album_image_url:
                st.image(album_image_url)

        with col2:
            # Display song name and artist
            st.markdown(f"**{rec['song_name']}**")
            st.markdown(f"{rec['artist_name']}")
            st.markdown(
                f"""
                <a href="{song_url}" target="_blank" style="
                    display: inline-block;
                    padding: 10px 20px;
                    font-size: 16px;
                    color: black;
                    background-color: #1DB954;
                    border: none;
                    border-radius: 25px;
                    text-decoration: none;
                    text-align: center;
                    cursor: pointer;
                ">
                    🎵 Listen on Spotify
                </a>
                """,
                unsafe_allow_html=True
            )

class SongRecommendationModel:
    def __init__(self, df_with_clusters, features):
        """
        Initialize the recommendation model with the data frame and features.
        df_with_clusters should contain the songs data with features and cluster labels.
        features should be a list of column names representing song features.
        """
        self.df = df_with_clusters
        self.features = features
        
        # Prepare feature matrix for clustering
        self.X = self.df[features].values
        
        # Fit KNN models for Cosine and Euclidean similarity
        self.knn_cosine = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn_cosine.fit(self.X)
        
        self.knn_euclidean = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn_euclidean.fit(self.X)

    def recommend_songs(self, input_track_id, max_neighbors=50):
        """
        Given a track ID, recommend unique overlapping songs using Cosine Similarity and Euclidean Distance.
        Ensures 5 unique overlapping songs are returned based on Track_Id and Track_Name, and sorts them by Track_Popularity.

        Parameters:
            input_track_id (str): The track ID of the input song.
            max_neighbors (int): The maximum number of neighbors to calculate for similarity. Defaults to 50.

        Returns:
            list: A list of up to 5 unique overlapping recommendations in the form of dictionaries.
        """
        # Get the input song
        input_song = self.df[self.df['Track_Id'] == input_track_id]
        
        if input_song.empty:
            return f"Track with ID {input_track_id} not found."
        
        # Get the input song's features
        input_song_features = input_song[self.features].values
        input_song_name = input_song['Track_Name'].iloc[0]
        
        # Get the cluster of the input song
        input_song_cluster = input_song['Cluster'].iloc[0]
        
        # Get songs from the same cluster
        songs_in_same_cluster = self.df[self.df['Cluster'] == input_song_cluster].reset_index(drop=True)
        
        if songs_in_same_cluster.empty:
            return f"No songs found in cluster {input_song_cluster}."
        
        # Feature matrix for songs in the same cluster
        X_cluster = songs_in_same_cluster[self.features].values
        
        # Initialize variables
        found_overlaps = []
        n_neighbors = 6  # Start with 6 neighbors (skipping the input song itself)
        
        # Iterate until we find at least 5 unique overlapping songs or reach max_neighbors
        while len(found_overlaps) < 5 and n_neighbors <= max_neighbors:
            # Find nearest neighbors using Cosine Similarity
            knn_cosine_cluster = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            knn_cosine_cluster.fit(X_cluster)
            cosine_neighbors = knn_cosine_cluster.kneighbors(input_song_features)
            
            # Find nearest neighbors using Euclidean Distance
            knn_euclidean_cluster = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            knn_euclidean_cluster.fit(X_cluster)
            euclidean_neighbors = knn_euclidean_cluster.kneighbors(input_song_features)
            
            # Extract Track IDs from Cosine and Euclidean results
            cosine_ids = set(
                songs_in_same_cluster.iloc[cosine_neighbors[1][0][1:]]['Track_Id'].values
            )  # Skip the input song
            euclidean_ids = set(
                songs_in_same_cluster.iloc[euclidean_neighbors[1][0][1:]]['Track_Id'].values
            )  # Skip the input song
            
            # Find unique overlapping Track IDs
            overlapping_ids = list(cosine_ids & euclidean_ids)
            
            # Filter unique recommendations based on Track_Id and Track_Name
            new_overlaps = [
                {"song_id": track_id,
                "song_name": songs_in_same_cluster[songs_in_same_cluster['Track_Id'] == track_id]['Track_Name'].iloc[0],
                "artist_name": songs_in_same_cluster[songs_in_same_cluster['Track_Id'] == track_id]['Artist_Name'].iloc[0],
                "track_popularity": songs_in_same_cluster[songs_in_same_cluster['Track_Id'] == track_id]['Track_Popularity'].iloc[0],
                "album_id": songs_in_same_cluster[songs_in_same_cluster['Track_Id'] == track_id]['Album_Id'].iloc[0]}  # Add album_id
                for track_id in overlapping_ids
                if track_id not in [rec["song_id"] for rec in found_overlaps] and
                songs_in_same_cluster[songs_in_same_cluster['Track_Id'] == track_id]['Track_Name'].iloc[0] not in [rec["song_name"] for rec in found_overlaps] and
                track_id != input_track_id and  # Exclude the input song by Track_Id
                songs_in_same_cluster[songs_in_same_cluster['Track_Id'] == track_id]['Track_Name'].iloc[0] != input_song_name  # Exclude the input song by Track_Name
            ]
            
            # Add to found overlaps
            found_overlaps.extend(new_overlaps)
            
            # Increase neighbors count for the next iteration
            n_neighbors += 5
        
        # Ensure no duplicates in Track_Id or Track_Name
        unique_overlaps = []
        seen_ids = set()
        seen_names = set()
        for overlap in found_overlaps:
            if overlap['song_id'] not in seen_ids and overlap['song_name'] not in seen_names:
                unique_overlaps.append(overlap)
                seen_ids.add(overlap['song_id'])
                seen_names.add(overlap['song_name'])
            
            # Stop when we have 5 unique recommendations
            if len(unique_overlaps) >= 6:
                break
        
        # Sort by Track_Popularity in descending order
        unique_overlaps = sorted(unique_overlaps, key=lambda x: x['track_popularity'], reverse=True)
        
        # Return only the first 6 unique overlapping songs (as dictionaries)
        return unique_overlaps[:6]

# Authentication flow
if "access_token" not in st.session_state:
    st.session_state.access_token = None

if not st.session_state.access_token or not is_token_valid(st.session_state.access_token):
    st.title("🎼Discover Your Next Favorite Song🎵")

    # Step 1: Display the Spotify login link
    auth_url = get_authorization_url()
    st.markdown(
                f"""
                <a href="{auth_url}" target="_blank" style="
                    display: inline-block;
                    padding: 10px 20px;
                    font-size: 16px;
                    color: black;
                    background-color: #1DB954;
                    border: none;
                    border-radius: 25px;
                    text-decoration: none;
                    text-align: center;
                    cursor: pointer;
                ">
                    🔐 Authorize Spotify
                </a>
                """,
                unsafe_allow_html=True
            )

    # Step 2: Handle callback with code and get token
    query_params = st.query_params
    if "code" in query_params:
        auth_code = query_params["code"]
        access_token = get_access_token(auth_code)

        if access_token:
            st.session_state.access_token = access_token
            st.success("Successfully authenticated with Spotify!")
            st.rerun()  # Reload to continue to the recommendation section
else:
    access_token = st.session_state.access_token
    
    # Load the recommendation model after successful authentication
    with open('song_recommendation_model_2.pkl', 'rb') as file:
        recommender = pickle.load(file)

    # Streamlit UI
    st.title('🎼Discover Your Next Favorite Song🎵')

    # Search and select field
    search_query = st.selectbox(
        "Search for a song to get recommendations:",
        options=[""] + [
            f"{row['Track_Name']} - {row['Artist_Name']}" 
            for _, row in recommender.df[['Track_Name', 'Artist_Name', 'Track_Id']].drop_duplicates().iterrows()
        ],
        format_func=lambda x: "Type to search" if x == "" else x,
    )

    if search_query and search_query != "Type to search":
        selected_track = recommender.df[  # Get the selected song's details
            (recommender.df['Track_Name'] + " - " + recommender.df['Artist_Name']) == search_query
        ].iloc[0]
        selected_track_id = selected_track['Track_Id']
        album_id = selected_track['Album_Id']

        # Recommendation button and display cards
        if st.button('Get Recommendations'):
            recommendations = recommender.recommend_songs(selected_track_id)
            if isinstance(recommendations, list):
                st.write("**Recommended Songs:**")

                # First Row (3 columns for 3 cards)
                cols = st.columns(3)  # Create 3 columns for the first row

                for idx, rec in enumerate(recommendations[:3]):  # First 3 songs
                    with cols[idx]:
                        display_song_card(rec, access_token)

                # Second Row (2 columns for 2 cards)
                cols = st.columns(3)  # Create 2 columns for the second row

                for idx, rec in enumerate(recommendations[3:], start=3): 
                    with cols[idx - 3]:  # Adjusting index for the second row
                        display_song_card(rec, access_token)

            else:
                st.write(recommendations)
