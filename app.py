import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("💻 Laptop Price Predictor")

@st.cache_data
def load_data():
    # ✅ Skip repeated header row (first row is duplicate in your CSV)
    df = pd.read_csv("laptop_data.csv", skiprows=1)
    df.columns = df.columns.str.strip()

    # ✅ Drop unwanted index column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # ✅ Remove any rows without 'GB' in Ram column
    df = df[df['Ram'].str.contains('GB', na=False)]

    # ✅ Drop any missing values
    df.dropna(inplace=True)

    # ✅ Clean 'Ram' column safely
    df['Ram'] = df['Ram'].astype(str).str.extract('(\d+)').astype(float).astype('Int64')

    # ✅ Clean 'Weight' column if it's a string
    if df['Weight'].dtype == 'object':
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)

    # ✅ Encode categorical columns
    label_encoders = {}
    categorical_cols = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

# ✅ Load dataset
df, encoders = load_data()

# ✅ Features and Target
X = df.drop('Price', axis=1)
y = df['Price']

# ✅ Train model
model = RandomForestRegressor()
model.fit(X, y)

st.header("📥 Enter Laptop Specifications")

# ✅ User Inputs
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

# ✅ Predict Price
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
