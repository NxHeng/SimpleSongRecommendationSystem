import streamlit as st
from linear_regression_app import run_linear_regression_app

st.set_page_config(page_title="Popularity Prediction", layout="wide")

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