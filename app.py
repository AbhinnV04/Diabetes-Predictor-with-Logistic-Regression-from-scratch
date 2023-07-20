import streamlit as st 
from predict import predict_page
from explore import show_explore_page

page = st.sidebar.selectbox("Explore Or Predict", ('Predict', 'Explore'))


if page == "Predict":
    predict_page()
else:
    show_explore_page()