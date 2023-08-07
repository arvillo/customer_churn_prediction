import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Page: ',('EDA','Predict Customer Churn'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()