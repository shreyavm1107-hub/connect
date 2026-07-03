import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set up page styling
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# --- App Branding Header ---
st.title("🛍️ Shopper Spectrum: E-Commerce Analytics Dashboard")
st.subheader("Created by Shreya Mohite")
st.write("An end-to-end Machine Learning pipeline for Customer Segmentation and Product Recommendations.")
st.markdown("---")

# --- Data & Machine Learning Pipeline ---
@st.cache_resource
def initialize_ml_pipeline():
    # ⚠️ PASTE YOUR COPIED RAW GITHUB CSV URL BETWEEN THE QUOTES BELOW:
    DATA_URL = "PASTE_YOUR_RAW_GITHUB_LINK_HERE"
    
    if DATA_URL == "PASTE_YOUR_RAW_GITHUB_LINK_HERE":
        st.error("🚨 Configuration Error: Please replace the placeholder link in app.py with your raw GitHub CSV URL.")
        st.stop()
        
    # Load data dynamically from your repository
    try:
        df = pd.read_csv(DATA_URL, encoding='ISO-8859-1')
    except Exception:
        df = pd.read_csv(DATA_URL, encoding='utf-8', errors='ignore')
        
    # Match uniform column formats
    df.columns = [col.strip() for col in df.columns]
    
    # Standard Data Cleaning
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    
    # Handle explicit date parsing variations
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate'], inplace=True)
    
    # Feature Engineering (RFM Analysis)
    snapshot_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'})
    
    rfm = rfm[(rfm['Recency'] > 0) & (rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]
    rfm_log = np.log1p(rfm)
    
    # Train Clustering Model
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    
    # Train Recommendation Engine (Filtered to optimize computing speeds)
    df['Description'] = df['Description'].str.strip()
    df = df[df['Description'] != ""]
    top_items = df['Description'].value_counts().head(400).index
    df_filtered = df[df['Description'].isin(top_items)]
    
    pivot_matrix = df_filtered.groupby(['Description', 'CustomerID'])['Quantity'].sum().unstack().fillna(0)
    item_similarity = cosine_similarity(pivot_matrix)
    similarity_df = pd.DataFrame(item_similarity, index=pivot_matrix.index, columns=pivot_matrix.index)
    
    return scaler, kmeans, similarity_df

# Run the pipeline
scaler, kmeans, similarity_df = initialize_ml_pipeline()

# Define customer group categories
CLUSTER_MAP = {
    0: {"name": "At-Risk / Churned Customers", "strategy": "Offer exclusive win-back discounts and send targeted 'We miss you' emails.", "color": "🔴"},
    1: {"name": "High-Value Champions", "strategy": "Offer early access to new collections and invite them to an elite loyalty rewards program.", "color": "🟢"},
    2: {"name": "New / Occasional Buyers", "strategy": "Send welcoming onboarding guides and initial purchase coupons.", "color": "🔵"},
    3: {"name": "Regular / Loyal Shoppers", "strategy": "Upsell related high-margin items and provide bundle milestones.", "color": "🟡"}
}

# --- Layout Configuration ---
tab1, tab2 = st.tabs(["🎯 Customer Segmentation Module", "📦 Product Recommendation Engine"])

# --- TAB 1: CUSTOMER SEGMENTATION ---
with tab1:
    st.header("Predict Customer Lifetime Behavior Segments")
    st.write("Input custom parameters below to evaluate an individual customer's business value profile.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Days since last checkout)", min_value=1, max_value=365, value=30)
    with col2:
        frequency = st.number_input("Frequency (Total distinct invoices)", min_value=1, max_value=500, value=5)
    with col3:
        monetary = st.number_input("Monetary Value (Total revenue generated ($))", min_value=1.0, max_value=50000.0, value=350.0)

    if st.button("Run Profile Classification"):
        input_data = np.array([[recency, frequency, monetary]])
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        
        predicted_cluster = kmeans.predict(input_scaled)[0]
        persona = CLUSTER_MAP[predicted_cluster]
        
        st.markdown("### **Profile Analysis Results:**")
        st.info(f"**Identified Category:** {persona['color']} **{persona['name']}**")
        st.success(f"**Recommended Marketing Plan:** {persona['strategy']}")

# --- TAB 2: PRODUCT RECOMMENDATIONS ---
with tab2:
    st.header("Product Recommendation Module")
    st.write("Select a retail product below to view complementary merchandise recommended by collaborative filtering filters.")
    
    product_list = similarity_df.index.tolist()
    selected_product = st.selectbox("Search or choose a retail catalog item:", product_list)
    
    if st.button("Generate Smart Recommendations"):
        if selected_product:
            recommendations = similarity_df[selected_product].sort_values(ascending=False).iloc[1:6]
            
            st.markdown("### Top 5 Frequently Bought Together Items:")
            cols = st.columns(5)
            for i, (prod_name, score) in enumerate(recommendations.items()):
                with cols[i]:
                    st.metric(label=f"Match #{i+1}", value=f"{score*100:.1f}% Confidence")
                    st.write(f"**{prod_name}**")
