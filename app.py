import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("💻 Laptop Price Predictor")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/VedPathak/Desktop/New folder/laptop_data.csv")

    # 🛠️ Fix: Drop index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df.dropna(inplace=True)

    # Clean RAM and Weight
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    if df['Weight'].dtype == 'object':
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    # Label Encoding
    label_encoders = {}
    for col in ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


    # Clean RAM and Weight
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    if df['Weight'].dtype == 'object':
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    # Encode categorical columns
    label_encoders = {}
    for col in ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

# Load the data and encoders
df, encoders = load_data()

# Split features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# UI for user input
st.header("📥 Enter Laptop Specifications")

# Use classes_ from each encoder for dropdowns
company = st.selectbox("Company", encoders['Company'].classes_)
typename = st.selectbox("Type", encoders['TypeName'].classes_)
inches = st.slider("Inches", 10.0, 20.0, 15.6)
screen_res = st.selectbox("Screen Resolution", encoders['ScreenResolution'].classes_)
cpu = st.selectbox("CPU", encoders['Cpu'].classes_)
ram = st.slider("RAM (in GB)", 2, 64, 8)
memory = st.selectbox("Memory", encoders['Memory'].classes_)
gpu = st.selectbox("GPU", encoders['Gpu'].classes_)
os = st.selectbox("Operating System", encoders['OpSys'].classes_)
weight = st.number_input("Weight (in Kg)", min_value=0.5, max_value=5.0, value=2.0)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[
        encoders['Company'].transform([company])[0],
        encoders['TypeName'].transform([typename])[0],
        inches,
        encoders['ScreenResolution'].transform([screen_res])[0],
        encoders['Cpu'].transform([cpu])[0],
        ram,
        encoders['Memory'].transform([memory])[0],
        encoders['Gpu'].transform([gpu])[0],
        encoders['OpSys'].transform([os])[0],
        weight
    ]])

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Laptop Price: ₹{int(prediction):,}")
