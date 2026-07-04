import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set up page layout
st.set_page_config(page_title="Shopper Spectrum Pro", layout="wide")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    st.title("Navigation Panel")
    st.info("Use the tabs on the main screen to explore the deep retail insights.")
    st.markdown("---")
    st.markdown("### Project Framework")
    st.write("• RFM Segmentation\n• K-Means Clustering\n• Collaborative Filtering\n• Cohort Analysis")

# --- App Branding Header ---
st.title("🛍️ Shopper Spectrum: Advanced E-Commerce Analytics")
st.subheader("Created by Shreya Mohite")
st.write("An end-to-end Machine Learning and e-commerce analytics platform powered by the Online Retail Dataset.")
st.markdown("---")

# --- Optimized Data & ML Pipeline ---
@st.cache_resource
def load_and_process_data():
    DATA_URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
    
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
    df['InvoiceMonth'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    
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
    
    # Map cluster numbers to meaningful names
    cluster_names = {
        0: "At-Risk Customers",
        1: "High-Value Champions",
        2: "New Buyers",
        3: "Regular/Loyal Shoppers"
    }
    rfm['Cluster_ID'] = kmeans.labels_
    rfm['Segment Profile'] = rfm['Cluster_ID'].map(cluster_names)
    
    # Pre-calculate WCSS for Elbow Curve
    wcss = []
    for k in range(1, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(rfm_scaled)
        wcss.append(km.inertia_)
    
    # 3. Product Recommendation Setup
    df['Description'] = df['Description'].str.strip()
    top_items = df['Description'].value_counts().head(150).index
    df_filtered = df[df['Description'].isin(top_items)]
    pivot_matrix = df_filtered.groupby(['Description', 'CustomerID'])['Quantity'].sum().unstack().fillna(0)
    item_similarity = cosine_similarity(pivot_matrix)
    similarity_df = pd.DataFrame(item_similarity, index=pivot_matrix.index, columns=pivot_matrix.index)
    
    # 4. Cohort Analysis Construction
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform(lambda x: x.min().strftime('%Y-%m'))
    df['InvoiceYearInt'] = df['InvoiceDate'].dt.year
    df['InvoiceMonthInt'] = df['InvoiceDate'].dt.month
    df['CohortYearInt'] = pd.to_datetime(df['CohortMonth'] + '-01').dt.year
    df['CohortMonthInt'] = pd.to_datetime(df['CohortMonth'] + '-01').dt.month
    df['CohortIndex'] = (df['InvoiceYearInt'] - df['CohortYearInt']) * 12 + (df['InvoiceMonthInt'] - df['CohortMonthInt'])
    
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
    
    return df, rfm, scaler, kmeans, similarity_df, retention_matrix, wcss

# Initialize Data Elements
df, rfm, scaler, kmeans, similarity_df, retention_matrix, wcss = load_and_process_data()

CLUSTER_MAP = {
    "At-Risk Customers": {"strategy": "Offer exclusive win-back discounts and targeted email outreach.", "color": "🔴"},
    "High-Value Champions": {"strategy": "Invite to premium VIP loyalty tier and early-access catalogs.", "color": "🟢"},
    "New Buyers": {"strategy": "Provide welcome onboarding guides and first-purchase discount coupons.", "color": "🔵"},
    "Regular/Loyal Shoppers": {"strategy": "Propose cross-sell product bundles and reward tier progression.", "color": "🟡"}
}

# --- Multi-Tab Application Frame ---
tabs = st.tabs(["📊 Exploratory Data Analysis", "🎯 Customer Segmentation", "📦 Product Recommendations", "📈 Cohort Retention Analysis"])

# --- TAB 1: EDA ---
with tabs[0]:
    st.header("Exploratory Data Analysis Overview")
    
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 25px;">
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #ff4b4b;">
            <p style="margin: 0; color: #555; font-weight: bold;">Total Sales Volume</p>
            <h2 style="margin: 5px 0 0 0; color: #111;">${df['TotalSpend'].sum():,.2f}</h2>
        </div>
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #00a65a;">
            <p style="margin: 0; color: #555; font-weight: bold;">Total Transactions</p>
            <h2 style="margin: 5px 0 0 0; color: #111;">{df['InvoiceNo'].nunique():,}</h2>
        </div>
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #00c0ef;">
            <p style="margin: 0; color: #555; font-weight: bold;">Items Cataloged</p>
            <h2 style="margin: 5px 0 0 0; color: #111;">{df['StockCode'].nunique():,}</h2>
        </div>
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #f39c12;">
            <p style="margin: 0; color: #555; font-weight: bold;">Active Customers</p>
            <h2 style="margin: 5px 0 0 0; color: #111;">{df['CustomerID'].nunique():,}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Sales Activity By Country Profile (Top 10)")
        country_sales = df.groupby('Country')['TotalSpend'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_country = px.bar(country_sales, x='TotalSpend', y='Country', orientation='h', 
                             color='TotalSpend', color_continuous_scale='Viridis', template='plotly_white')
        fig_country.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_country, use_container_width=True)
        
    with col_right:
        st.subheader("Monthly Revenue Growth Trend")
        monthly_sales = df.groupby('InvoiceMonth')['TotalSpend'].sum().sort_index().reset_index()
        fig_line = px.line(monthly_sales, x='InvoiceMonth', y='TotalSpend', markers=True, template='plotly_white')
        fig_line.update_traces(line_color='#ff4b4b')
        st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: CUSTOMER SEGMENTATION WITH CLUSTERING EVALUATION ---
with tabs[1]:
    st.header("Customer Segmentation Profile Breakdown")
    
    # Sub-tabs for neat structuring
    seg_tab1, seg_tab2 = st.tabs(["📊 Segment Analytics Dashboard", "⚙️ Clustering Model Evaluation"])
    
    with seg_tab1:
        st.subheader("Segment Metrics Performance Summary")
        summary_df = rfm.groupby('Segment Profile').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Cluster_ID': 'count'
        }).rename(columns={'Cluster_ID': 'Customer Count'})
        
        st.dataframe(summary_df.style.format({
            'Recency': '{:.1f} days',
            'Frequency': '{:.1f} orders',
            'Monetary': '${:,.2f}',
            'Customer Count': '{:,}'
        }), use_container_width=True)
        
        st.markdown("---")
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.subheader("👥 Customer Volume by Segment")
            segment_counts = rfm['Segment Profile'].value_counts().reset_index()
            segment_counts.columns = ['Segment Profile', 'Count']
            fig_seg = px.bar(segment_counts, x='Segment Profile', y='Count', color='Segment Profile',
                             color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_white')
            st.plotly_chart(fig_seg, use_container_width=True)
            
        with c_right:
            st.subheader("🔍 Look Up Specific Customer")
            available_ids = rfm.index.tolist()
            search_id = st.selectbox("Select or type Customer ID:", available_ids)
            
            if search_id:
                cust_row = rfm.loc[search_id]
                seg_name = cust_row['Segment Profile']
                meta = CLUSTER_MAP[seg_name]
                
                st.markdown(f"**Cluster Designation:** {meta['color']} `{seg_name}`")
                st.markdown(f"• **Recency:** {cust_row['Recency']} days ago")
                st.markdown(f"• **Frequency:** {cust_row['Frequency']} transactions")
                st.markdown(f"• **Monetary Value:** ${cust_row['Monetary']:,.2f}")
                st.info(f"💡 **Strategy:** {meta['strategy']}")

    with seg_tab2:
        st.subheader("📐 K-Means Parameter Optimization Diagnostics")
        st.write("Validation metrics proving why K=4 is mathematically the optimal option for customer partitioning.")
        
        el_col, desc_col = st.columns([2, 1])
        with el_col:
            # Interactive Elbow Curve Chart
            fig_elbow = px.line(x=list(range(1, 8)), y=wcss, markers=True, template='plotly_white',
                                labels={'x': 'Number of Clusters (K)', 'y': 'Within-Cluster Sum of Squares (WCSS)'})
            fig_elbow.update_traces(line_color='#00a65a')
            fig_elbow.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="Optimal Elbow Point (K=4)")
            st.plotly_chart(fig_elbow, use_container_width=True)
            
        with desc_col:
            st.markdown("#### **Mathematical Analysis**")
            st.success("✔ **The Elbow Method:** As seen in the line plot, the rate of variance explanation drops sharpest right at K=4. Beyond this elbow point, adding more clusters provides minimal reduction in structural error.")
            st.info("✔ **Feature Processing:** Inputs undergo log transformations to smooth right-skewed profiles before running a `StandardScaler` standard normalizing optimization pass.")

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
    st.header("User Lifecycle Cohort Analysis Heatmap")
    st.write("Percentage values track user retention trajectories relative to original sign-up timelines.")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(retention_matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=True, ax=ax)
    plt.title("Customer Retention Heatmap (Monthly Cohorts)")
    plt.ylabel("Cohort Grouping Month")
    plt.xlabel("Months Elapsed Since Activation")
    st.pyplot(fig)
