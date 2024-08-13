#We need to load the pixkled files because we need them for prediction. First import the necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('onehot_encoder_gender.pkl', 'rb') as file:
     one_hot_encoder_gen= pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
     one_hot_encoder_geo= pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn prediction')

#User_input 
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', one_hot_encoder_gen.categories_[0])
age = st.slider('Age', 18, 92)
balance = st.slider('Balance', min_value=0, max_value=260000, step=1)  
credit_Score = st.slider('CreditScore', min_value=350, max_value=850, step=1)
Estimated_Salary = st.slider('EstimatedSalary', min_value=11.58, max_value=199992.48)
tenure = st.slider('Tenure', min_value=0, max_value=10)
num_of_products=st.slider('NumOfProducts', 1, 4)
has_cr_Card = st.selectbox('HasCrCard',[0,1])
is_active_member = st.selectbox('IsActiveMember',[0,1])

input_data = pd.DataFrame({
        'CreditScore' : [credit_Score], 
        'Geography' : [geography],
        'Gender' : [gender],
        'Age' : [age],
        'Tenure' : [tenure],
        'Balance' : [balance],
        'NumOfProducts' : [num_of_products], 
        'HasCrCard' :[has_cr_Card ],
        'IsActiveMember' : [is_active_member], 
        'EstimatedSalary': [Estimated_Salary] }
)

geo_encoder=one_hot_encoder_geo.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

gen_encoder=one_hot_encoder_gen.transform(input_data[['Gender']])
gen_encoded_df = pd.DataFrame(gen_encoder, columns=one_hot_encoder_gen.get_feature_names_out(['Gender']))
# Drop the original categorical columns
input_data = input_data.drop(['Geography', 'Gender'], axis=1)

input_data = pd.concat([input_data,gen_encoded_df,geo_encoded_df], axis=1)

scaled_input = scaler.transform(input_data)


# # Make prediction
prediction = model.predict(scaled_input)
prediction_prob=prediction[0][0]
st.write(f'Churn probability: {prediction_prob:.2f}')

if (prediction_prob > 0.5):
   st.write('The person is likely to churn')
else:
  st.write('The person is not likely to churn')
