import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load encoders and scaler
with open('onehot_en_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_en_gen.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit app
st.title('Customer Churn Prediction')

st.write('Enter customer details to predict whether the customer will leave the bank.')

# Take input from user
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=300)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Predict button
if st.button('Predict'):

    # Convert gender into numerical value
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]])

    # Convert sparse matrix into array if required
    if hasattr(geo_encoded, 'toarray'):
        geo_encoded = geo_encoded.toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine all input features
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    # Arrange columns according to training data
    if hasattr(scaler, 'feature_names_in_'):
        input_data = input_data[scaler.feature_names_in_]

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    prediction_probability = prediction[0][0]

    # Display result
    st.subheader('Prediction Result')

    st.write('Churn Probability:', round(float(prediction_probability) * 100, 2), '%')

    if prediction_probability > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
