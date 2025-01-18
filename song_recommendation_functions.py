import streamlit as st
import requests
import base64
import urllib.parse
import os

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