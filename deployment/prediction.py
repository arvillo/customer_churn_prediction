import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
import numpy as np

# Load All Files
with open('preprocessor.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

model_ann = load_model('model_ann.h5')

def run():
    st.session_state.disabled = True
    user_id = st.text_input('User ID', max_chars=16, key='user_id')
    if len(user_id) != 16:
       st.warning('User ID must have 16 characters')
       st.session_state.disabled = True
    else:
      st.session_state.disabled = False
    login_today = st.radio('Did customer login today?',('Yes','No'), index=0, disabled=st.session_state.disabled)
    if login_today == 'Yes':
      days_since_last_login = -999
    else:
      days_since_last_login = st.slider('How Many days that customer last login?', min_value=0, max_value=30, value=0, step=1, help='Last Login Days')
    with st.form('key=form_customer_churn'):
      age = st.slider('How old is the customer', min_value=15, max_value=70, value=21, step=1, help='Customer Age',
                                      disabled=st.session_state.disabled)
      gender = st.radio('Did the customer female or male?',('Female','Male'), index=1, disabled=st.session_state.disabled)
      if gender == 'Female':
        gender = 'F'
      else:
        gender = 'M'
      region_category = st.selectbox('What Region Category Customer From?',
                                        ('City','Village','Town'),
                                        index=0, disabled=st.session_state.disabled)
      membership_category = st.selectbox('What Region Category Customer From?',
                                        ('No Membership', 'Basic Membership', 'Silver Membership',
                                         'Premium Membership', 'Gold Membership', 'Platinum Membership'),
                                        index=0, disabled=st.session_state.disabled)
      joining_date = st.date_input('Booking Date', disabled=st.session_state.disabled, value=datetime.today(),
                                   max_value=datetime.today())
      joined_through_referral = st.radio('Did the customer joined through referral?',('Yes','No'), index=0, disabled=st.session_state.disabled)
      preferred_offer_types = st.selectbox('What is Customer Preferred Offer Types?',
                                        ('Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'),
                                        index=0, disabled=st.session_state.disabled)
      medium_of_operation = st.selectbox('What is Customer Medium Of Operation?',
                                        ('Desktop', 'Smartphone', 'Both'),
                                        index=0, disabled=st.session_state.disabled)
      internet_option = st.selectbox('What is Customer Internet Option?',
                                        ('Wi-Fi', 'Fiber_Optic', 'Mobile_Data'),
                                        index=0, disabled=st.session_state.disabled)
      last_visit_time = st.time_input('When is the last time customer visit?',help='Time customer last visit')
      avg_time_spent = st.number_input('Average Time Spent', max_value= 36000.0
                                       ,step=1.,format="%.2f", disabled=st.session_state.disabled)
      avg_transaction_value = st.number_input('Average Transaction Value', min_value= 800.0, max_value=100000.0,
                                              step=1.,format="%.2f", disabled=st.session_state.disabled)
      avg_frequency_login_days = st.number_input('Average Frequency Login Days', min_value=0.0, max_value=100.0,
                                                 step=1.,format="%.2f", disabled=st.session_state.disabled)
      points_in_wallet = st.number_input('Points in Wallet', min_value=0.0, max_value=3000.0,
                                           step=1.,format="%.2f", disabled=st.session_state.disabled)
      used_special_discount = st.radio('Did the customer use special discount?',('Yes','No'), index=0, disabled=st.session_state.disabled)
      offer_application_preference = st.radio('Did the customer offer application preference?',('Yes','No'), index=0, disabled=st.session_state.disabled)
      past_complaint = st.radio('Did the customer has past complaint?',('Yes','No'), index=0, disabled=st.session_state.disabled)
      complaint_status = st.selectbox('What is Customer Complain Status?',
                                        ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'),
                                        index=0, disabled=st.session_state.disabled)
      feedback = st.selectbox('What is Customer Feedback?',
                              ('Poor Website', 'Poor Customer Service', 'Too many ads',
                               'Poor Product Quality', 'No reason specified',
                               'Products always in Stock', 'Reasonable Price',
                               'Quality Customer Care', 'User Friendly Website'),
                               index=0, disabled=st.session_state.disabled)
      submitted = st.form_submit_button('Predict', disabled=st.session_state.disabled)

    # Create New Data

    data_inf = {
      'user_id': user_id,
      'age': age,
      'gender': gender,
      'region_category': region_category,
      'membership_category': membership_category,
      'joining_date': joining_date,
      'joined_through_referral': joined_through_referral,
      'preferred_offer_types': preferred_offer_types,
      'medium_of_operation': medium_of_operation,
      'internet_option': internet_option,
      'last_visit_time': last_visit_time,
      'days_since_last_login': days_since_last_login,
      'avg_time_spent': avg_time_spent,
      'avg_transaction_value': avg_transaction_value,
      'avg_frequency_login_days': avg_frequency_login_days,
      'points_in_wallet': points_in_wallet,
      'used_special_discount': used_special_discount,
      'offer_application_preference': offer_application_preference,
      'past_complaint': past_complaint,
      'complaint_status': complaint_status,
      'feedback': feedback
    }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        data_inf_transform = model_pipeline.transform(data_inf)
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)  
        if y_pred_inf[0] == 0:
            st.write('# The customer is not churn')
        else:
            st.write('# The customer is churn')