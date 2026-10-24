import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")
st.title("SuperKart Sales Forecasting System")

# Default to Docker network service name 'backend:7860' inside Codespaces
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

# ── Online (Single) Prediction Section ──
st.header("Online Sales Prediction (Single Product)")

col1, col2 = st.columns(2)

with col1:
    Product_Weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.66, step=0.1)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area Ratio", min_value=0.001, max_value=0.500, value=0.027, step=0.005, format="%.3f")
    Product_MRP = st.number_input("Product MRP", min_value=10.0, max_value=350.0, value=117.08, step=1.0)
    Product_Id_char = st.selectbox("Product ID Prefix (FD=Food, NC=Non-Consumable, DR=Drinks)", ["FD", "NC", "DR"])

with col2:
    Store_Size = st.selectbox("Store Size", ["Medium", "High", "Small"])
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=50, value=16, step=1)
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

payload = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": int(Store_Age_Years),
    "Product_Type_Category": Product_Type_Category
}

if st.button("Predict Store Sales", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
    if response.status_code == 200:
        pred_val = response.json().get("Predicted_Store_Sales")
        st.success(f"Predicted Product Store Sales Total: **${pred_val:,.2f}**")
    else:
        st.error(f"Error ({response.status_code}): {response.text}")

st.divider()

# ── Batch Prediction Section ──
st.header("Batch Sales Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload Batch CSV file", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch Sales", type="primary"):
    response = requests.post(
        f"{BACKEND_URL}/v1/predictbatch",
        files={"file": uploaded_file.getvalue()}
    )
    if response.status_code == 200:
        st.success("Batch predictions completed successfully!")
        batch_preds = response.json()
        results_df = pd.DataFrame(list(batch_preds.items()), columns=["Row / Product ID", "Predicted Store Sales"])
        st.dataframe(results_df, use_container_width=True)
    else:
        st.error(f"Error ({response.status_code}): {response.text}")
