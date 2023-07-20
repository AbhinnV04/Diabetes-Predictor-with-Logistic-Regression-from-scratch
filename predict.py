import streamlit as st 
import pickle 
import numpy as np 

with open('data/model.pk1', 'rb') as file:
    data = pickle.load(file)      
model = data['model']

def predict_page():
    st.title("Diabetes Prediction Model")
    
    st.image('https://d2jx2rerrg6sh3.cloudfront.net/images/news/ImageForNews_733883_16711528404605542.jpg')
    
    st.write("""### We need Some information for prediction""")
    age = st.slider('Age', 0, 100, 50)
    pregnancies = st.slider('Number of Pregnancies', 0, 20, 6)
    bmi = st.slider('BMI', max_value = 60., min_value= 0., value=33.6 ,step=0.10)
    glucose = st.number_input('Glucose level(mg/dL)', 0, 300, 148)
    blood_pressure = st.number_input('Blood pressure', 0, 150, 72)
    skin_thickness = st.number_input('Skin thickness', 0, 120, 35)
    insulin = st.number_input('Insulin', 0, 850, 0)
    pedigree_function = st.number_input('Diabetes Pedigree Function', min_value = 0.,
                                        max_value = 2.5, step=0.01, value=0.63) 
    st.write("""
             *(Model type = 'Logistic Regression'*\n
             Model accuracy = 80%*)*
             """)
    calculate = st.button('Calculate')
    
    if calculate:
        X = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])
        X = X.astype(float)
        
        outcome = model.predict(X)
        if outcome == 1:
            outcome_output = 'Positive'
        else:
            outcome_output = 'Negative'
        st.subheader(f"Outcome: {outcome_output}")
