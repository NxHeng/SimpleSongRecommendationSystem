import streamlit as st

# Title of the app
st.set_page_config(page_title="About", layout="wide")
st.title("About the App")

# Description of the app
st.write("""
This application showcases two exciting machine learning projects:
1. **Linear Regression**: A model that predicts track popularity based on input data.
2. **Song Recommendation**: A system that suggests songs based on user preferences.

Please use the sidebar to navigate to the respective project and explore them in detail.
""")

# Optional: Add an image or some graphics related to the projects
st.image("https://wallpapers.com/images/hd/spotify-logo-green-background-721x5rev4byd7t68.jpg", caption="Machine Learning Projects with Spotify Web API", use_container_width=True)

# More information or call to action
st.write("""
Feel free to explore the projects and dive deeper into how they work. You can click on each project in the sidebar to learn more.
""")
