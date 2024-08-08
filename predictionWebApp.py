# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:51:10 2024

@author: Nirbhay Singh
"""

import numpy as np
import pickle
import streamlit as st

file_path = "C:/Users/Nirbhay Singh/OneDrive/Desktop/ML Project Disease Classification/trained_model.sav"
loaded_model = pickle.load(open(file_path, 'rb'))

def diabetesPred(input_data):
    # Ensure input_data is a list of floats
    input_data = [float(i) for i in input_data]

    # Convert the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Return the prediction result
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title('Diabetes Prediction WEB APP')
    
    # Input fields
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the person')
    
    diagnosis = ''
    
    if st.button('Test Result'):
        # Check if all inputs are valid numbers
        try:
            input_data = [
                Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                DiabetesPedigreeFunction, Age
            ]
            diagnosis = diabetesPred(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numeric values for all inputs.'
        
        # Display the result
        st.success(diagnosis)
        
if __name__ == '__main__':
    main()