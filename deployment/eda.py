import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chisquare

st.set_page_config(
    page_title='Hotel Reservation - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Title page
    st.title('Customer Churn Prediction')

    # Markdown 
    st.markdown('---')
    data = pd.read_csv('churn.csv')
    num_col = data.select_dtypes(include=np.number).columns[data.select_dtypes(include=np.number).notnull().all()].tolist()
    col_miss = ['preferred_offer_types','gender']
    cat_col = data.select_dtypes(include=['object']).columns[data.select_dtypes(include=['object']).notnull().all()].tolist()
    cat_col = np.concatenate((cat_col,col_miss,['churn_risk_score']), axis=0)
    cat_col = np.delete(cat_col, [0,2,4])

    # Countplot untuk data Categorical
    st.write('#### Countplot for categorical data')
    total_col = len(cat_col)
    fig = plt.figure(figsize=(16, total_col * 10))
    i = 1
    for col in cat_col:
        plt.subplot(total_col * 2, 2, i)
        sns.countplot(x = col, hue = 'churn_risk_score', palette = 'Set1', data = data)
        plt.xticks(rotation=45)
        plt.title(f'Countplot of {col}')
        i += 1

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Dari hasil analisa diatas, dapat dikatakan bahwa customer yang churn cenderung memiliki \
             **feedback yang negatif** kepada perusahaan dan cenderung **tidak memiliki membership \
             ataupun member basic** saja.')
    st.markdown('---')

    # KDE Plot untuk data Numerical
    st.write('#### KDE Plot of Numerical Data')
    total_col = len(num_col)
    fig = plt.figure(figsize=(16, total_col * 10))
    i = 1
    for col in num_col:
        if col != 'churn_risk_score':
            plt.subplot(total_col * 2, 2, i)
            sns.kdeplot(x = col, hue = 'churn_risk_score', palette = 'Set1', fill=True, data = data)
            plt.xticks(rotation=45)
            plt.title(f'KDE Plot of {col}')
            i += 1

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Dari hasil analisa diatas, customer yang melakukan churn cenderung memiliki \
             **rata - rata transaksi** yang **rendah** dibandingkan dengan customer yang \
             tidak churn')
    st.markdown('---')

    # Pie Chart untuk cek distribusi kelas
    st.write('#### Pie Chart for Churn Risk Score')
    fig = plt.figure(figsize=(15,5))
    data.groupby('churn_risk_score')['user_id'].count().plot.pie(y='user_id', autopct='%0.2f', figsize=(10,10))
    st.pyplot(fig)
    st.write('Distribusi pada data ini dapat dikatakan **balance** sehingga data tidak \
             perlu dilakukan **oversample** ataupun **undersample**')
    st.markdown('---')

    # Histogram untuk cek distribusi data
    st.write('#### Histogram to see if the data skewed or not')
    total_col = len(num_col)
    fig = plt.figure(figsize=(25, total_col * 10))
    i = 1
    for col in num_col:
        if col != 'churn_risk_score':
            plt.subplot(total_col * 4, 4, i)
            sns.histplot(data[col], kde=True, bins = 30)
            plt.title(f'Histogram of {col}')
            i += 1

    plt.tight_layout()
    st.pyplot(fig)

    st.write('Dari Visualisasi diatas dapat disimpulkan bahwa distribusi dibagi data menjadi 3 yaitu:')
    st.write('- Data **age** memiliki pemusatan data yang sudah terpusah ke tengah.')
    st.write('- Data **avg_time_spent**, **avg_transaction_value**, **avg_frequency_login_days** dan \
             **points_in_wallet** memiliki distribusi data yang cenderung condong ke kiri.')
    st.write('- Data **days_since_login** memiliki pemusatan data yang condong ke sebelah kanan')
    st.markdown('---')
    
    st.write('#### Boxplot to see data outlier')
    total_col = len(num_col)
    fig = plt.figure(figsize=(16, total_col * 10))
    i = 1
    for col in num_col:
        plt.subplot(total_col * 4, 4, i)
        sns.boxplot(x=data[col])
        plt.title(f'Boxplot of {col}')
        i += 1

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Dari hasil visualisasi boxplot, data **age** tidak memiliki outlier, \
             namun data - data lainnya memiliki outlier yang cukup banyak seperti:')
    st.write('- Data **days_since_login** memiliki outlier pada data bagian kiri dengan nilai cukup ekstrim')
    st.write('- Data **avg_time_spent** memiliki outlier pada data bagian kanan dengan nilai cukup ekstrim')
    st.write('- Data **avg_transaction_value** dan **avg_frequency_login_days** memiliki outlier pada bagian kanan')
    st.write('- Data **point_in_wallet** memiliki outlier di kedua sisi bagian data')
    st.markdown('---')

    # Heatmap untuk korelasi data
    st.write('#### Barchart for P-Value in T-test (Only Numerical with Churn Risk Score)')
    p_val_num = {}
    for col in num_col:
        if col != 'churn_risk_score':
            t_stat, p_val = ttest_ind(data[data['churn_risk_score'] == 0][col],data[data['churn_risk_score'] == 1][col])
            p_val_num[f'{col}'] = p_val
    fig = plt.figure(figsize=(10,10))
    pd.Series(p_val_num).sort_values(ascending=False).plot.bar()
    st.pyplot(fig)
    st.write('Dari hasil visualisasi dan uji statistik t-test diatas, dapat dikatakan bahwa data yang memiliki \
             hubungan signifikan dengan data **churn_risk_score** ialah **avg_time_spent**, **avg_frequency_login_days**\
             , **avg_transaction_value**, dan **point_in_wallet**')
    st.markdown('---')

    # Chi Test Result
    st.write('#### Barchart for P-Value in Chi Square Test (Only Categorical with Churn Risk Score)')
    p_val_cat = {}
    for col in num_col:
        if col != 'churn_risk_score':
            t_stat, p_val = chisquare(pd.crosstab(data[col], data['churn_risk_score']), axis=None)
            p_val_cat[f'{col}'] = p_val
    fig = plt.figure(figsize=(10,10))
    pd.Series(p_val_cat).sort_values(ascending=False).plot.bar()
    st.pyplot(fig)
    st.write('Dari hasil visualisasi dan uji statistik chi-square diatas, dapat dikatakan bahwa seluruh \
             data categorical memiliki hubungan signifikan dengan data **churn_risk_score**')
    st.markdown('---')
    