import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set up page layout
st.set_page_config(page_title="Shopper Spectrum Pro", layout="wide")

# --- App Branding Header ---
st.title("🛍️ Shopper Spectrum: Advanced E-Commerce Analytics")
st.subheader("Created by Shreya Mohite")
st.write("An end-to-end Machine Learning and Business Intelligence platform powered by the Online Retail Dataset.")
st.markdown("---")

# --- Optimized Data & ML Pipeline ---
@st.cache_resource
def load_and_process_data():
    # Explicitly defining the URL right at the top of the function
    DATA_URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
    
    # Fast load: Use only the first 150,000 rows to prevent the free server from stalling
    df = pd.read_csv(DATA_URL, encoding='ISO-8859-1', nrows=150000)
    df.columns = [col.strip() for col in df.columns]
    
    # Cleaning
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate'], inplace=True)
    
    # 1. Base Data for EDA
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    
    # 2. RFM Feature Engineering for Clustering
    snapshot_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'})
    rfm = rfm[(rfm['Recency'] > 0) & (rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]
    
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    
    # 3. Product Recommendation Setup (Optimized to top 150 items for speed)
    df['Description'] = df['Description'].str.strip()
    top_items = df['Description'].value_counts().head(150).index
    df_filtered = df[df['Description'].isin(top_items)]
    pivot_matrix = df_filtered.groupby(['Description', 'CustomerID'])['Quantity'].sum().unstack().fillna(0)
    item_similarity = cosine_similarity(pivot_matrix)
    similarity_df = pd.DataFrame(item_similarity, index=pivot_matrix.index, columns=pivot_matrix.index)
    
    # 4. Cohort Analysis Construction
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform(lambda x: x.min().to_period('M'))
    df['CohortIndex'] = (df['InvoiceMonth'].dt.year - df['CohortMonth'].dt.year) * 12 + (df['InvoiceMonth'].dt.month - df['CohortMonth'].dt.month)
    
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
    
    return df, rfm, scaler, kmeans, similarity_df, retention_matrix

# Initialize Data Elements
df, rfm, scaler, kmeans, similarity_df, retention_matrix = load_and_process_data()

CLUSTER_MAP = {
    0: {"name": "At-Risk Customers", "strategy": "Win-back discounts.", "color": "🔴"},
    1: {"name": "High-Value Champions", "strategy": "VIP Program invitation.", "color": "🟢"},
    2: {"name": "New Buyers", "strategy": "Welcome onboarding sequences.", "color": "🔵"},
    3: {"name": "Regular/Loyal Shoppers", "strategy": "Cross-sell bundle milestones.", "color": "🟡"}
}

# --- Multi-Tab Application Frame ---
tabs = st.tabs(["📊 Exploratory Data Analysis", "🎯 Customer Segmentation", "📦 Product Recommendations", "📈 Cohort Retention Analysis"])

# --- TAB 1: EDA ---
with tabs[0]:
    st.header("Exploratory Data Analysis Overview")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales Volume", f"${df['TotalSpend'].sum():,.2f}")
    m2.metric("Total Transactions", f"{df['InvoiceNo'].nunique():,}")
    m3.metric("Unique Items Cataloged", f"{df['StockCode'].nunique():,}")
    m4.metric("Active Customer Base", f"{df['CustomerID'].nunique():,}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Sales Activity By Country Profile (Top 10)")
        country_sales = df.groupby('Country')['TotalSpend'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(country_sales)
        
    with col_right:
        st.subheader("Monthly Sales Trend")
        monthly_sales = df.groupby('InvoiceMonth')['TotalSpend'].sum().sort_values(by='InvoiceMonth')
        monthly_sales.index = monthly_sales.index.astype(str)
        st.line_chart(monthly_sales)

# --- TAB 2: SEGMENTATION ---
with tabs[1]:
    st.header("Predictive Customer Classification Engine")
    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Days since last checkout", 1, 365, 45)
    with c2:
        frequency = st.number_input("Total checkout instances", 1, 500, 12)
    with c3:
        monetary = st.number_input("Total continuous revenue spending ($)", 1.0, 50000.0, 850.0)
        
    if st.button("Evaluate Customer Persona Group"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(np.log1p(input_data))
        cluster_id = kmeans.predict(input_scaled)[0]
        meta = CLUSTER_MAP[cluster_id]
        st.info(f"Classification: {meta['color']} **{meta['name']}**")
        st.success(f"Strategic Directing: {meta['strategy']}")

# --- TAB 3: RECOMMENDATIONS ---
with tabs[2]:
    st.header("Item Collaborative Recommendation Matrix")
    selected_prod = st.selectbox("Select a catalog merchandise asset:", similarity_df.index.tolist())
    if st.button("Run Similarity Matrix Matching"):
        recs = similarity_df[selected_prod].sort_values(ascending=False).iloc[1:6]
        cols = st.columns(5)
        for idx, (p_name, metric_score) in enumerate(recs.items()):
            with cols[idx]:
                st.metric(f"Match Rank #{idx+1}", f"{metric_score*100:.1f}%")
                st.write(p_name)

# --- TAB 4: COHORT ANALYSIS ---
with tabs[3]:
    st.header("User Lifecycle Cohort Analysis")
    st.write("Percentage values track user retention trajectories relative to original sign-up timelines.")
    
    retention_matrix.index = retention_matrix.index.astype(str)
    formatted_retention = retention_matrix.style.format("{:.1%}", na_rep="")
    st.dataframe(formatted_retention, use_container_width=True)
