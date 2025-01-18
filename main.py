import streamlit as st
from linear_regression_app import run_linear_regression_app

# Title of the app
st.title("Machine Learning Projects")

# Create a navigation bar
project_option = st.radio(
    "Select the project you want to explore:",
    ("Linear Regression", "Song Recommendation")
)

# Linear Regression Project
if project_option == "Linear Regression":
    st.header("Linear Regression Model")
    st.write("Here, you can display the details, graphs, or predictions for your Linear Regression model.")
    run_linear_regression_app()

    # If you want to load and display some specific outputs (like graphs or predictions) from the LinearRegression model:
    # Example:
    # st.write("Model Coefficients:", model.coef_)
    # st.write("Predictions:", predictions)

    # Add more functionality specific to your Linear Regression model
    # For instance, you can display a plot or use the model for predictions:
    # st.line_chart(linear_regression_data)

    # You could also include a file upload option or other Streamlit widgets
    # that interact with the LinearRegression model to make predictions.

# Song Recommendation Project
elif project_option == "Song Recommendation":
    st.header("Song Recommendation Model")
    st.write("Here, you can display the details, song recommendations, or results for your Song Recommendation model.")
    
    # If you want to load and display song recommendations or predictions:
    # Example:
    # st.write("Recommended Songs:", song_recommendations)
    # st.write("Artist Popularity Prediction:", artist_popularity)

    # Add more functionality specific to the SongRecommendation model
    # For instance, you could show song recommendations based on user input:
    # st.write("Enter song features to get recommendations:")

    # Use relevant code from your SongRecommendation model to interact with users.

