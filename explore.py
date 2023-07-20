import streamlit as st 
import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv('data/diabetes.csv')
    return df

df = load_data()

def load_bmi():
    conditions = [(df['BMI'] >= 25.0),
              (df['BMI'] <= 18.5),
              (df['BMI'] > 18.5) & (df['BMI'] < 25.0)]

    choices = ['over', 'under', 'normal']

    df_bmi = df[['BMI', 'Outcome']]
    df_bmi['BMI_category'] = np.select(conditions, choices, default='normal')
    return df_bmi

def show_explore_page():
    st.title('Exploring Diabetes Dataset')
    st.write('Link to Dataset: https://www.kaggle.com/datasets/kandij/diabetes-dataset')
    
    st.image('https://files.realpython.com/media/Intro-to-Exploratory-Data-Analysis-With-Pandas_Watermarked.81a7d7df468f.jpg')
    
    st.write("#### A sample of the training dataset")
    st.table(df.head(10))
    rows, col = df.shape
    st.write(f"""size -> {rows} x {col}""")
    
    st.write("#### Positive and Negative Outcomes")
    p_v_n = plt.figure(figsize=(6,3))
    sns.countplot(x='Outcome', data= df)
    st.pyplot(p_v_n)    
    
    st.write("#### Outcome and Age")
    age = plt.figure(figsize=(22,8)) 
    sns.countplot(x='Age', hue='Outcome',data=df)
    st.pyplot(age)
    
    st.write("#### Outcome and BMI")
    df_bmi = load_bmi()
    bmi = plt.figure(figsize=(22,8)) 
    sns.countplot(x='BMI_category', hue='Outcome', data=df_bmi)
    st.pyplot(bmi)
    
    st.write("#### BMI and Age")
    bmi_age = plt.figure(figsize=(12,5))
    sns.boxplot(x='Age', y='BMI', data=df[df['Age']<61])
    st.pyplot(bmi_age)
    
    
    st.write("#### Correlation Chart")
    corr = plt.figure(figsize=(8,6))
    sns.heatmap(df.corr())
    st.pyplot(corr)
    
    st.write("""
             Obseravations -

            - Younger People (21-30) have more negative results.
            - Overweight individuals have the highest rate of positive diagnosis.
            - Glucose has the highest correlation factor thus is the major predictor for the outcome, followed by BMI.
             """)
