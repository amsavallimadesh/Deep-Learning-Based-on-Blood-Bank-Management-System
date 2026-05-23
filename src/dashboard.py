import streamlit as st
import pandas as pd

st.title("BloodNet AI – Blood Demand Forecast")

uploaded = st.file_uploader("Upload prediction CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Demand Trend")
    st.line_chart(df.select_dtypes(include="number"))

