import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# -----------------------------
# [1] DATA LOADING & PREP
# -----------------------------
@st.cache_data

def load_data(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df.drop_duplicates(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    return df

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")
st.title("🛒 E-commerce Customer Segmentation Dashboard")

file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
if file:
    df = load_data(file)

    # -----------------------------
    # [2] EXPLORATORY DATA ANALYSIS
    # -----------------------------
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x='TotalSales', nbins=50, title="Total Sales Distribution"))
    with col2:
        st.plotly_chart(px.histogram(df, x='Quantity', nbins=50, title="Quantity Distribution"))

    st.plotly_chart(px.scatter(df, x='Quantity', y='TotalSales', color='CustomerID', title="Quantity vs Total Sales"))

    # -----------------------------
    # [3] RFM ANALYSIS
    # -----------------------------
    st.header("📦 RFM Analysis")
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSales': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Tenure'] = (snapshot_date - df.groupby('CustomerID')['InvoiceDate'].min()).dt.days
    df['BasketSize'] = df.groupby('InvoiceNo')['StockCode'].transform('count')
    rfm['AvgBasketSize'] = df.groupby('CustomerID')['BasketSize'].mean()
    st.dataframe(rfm.describe())

    # -----------------------------
    # [4] CLUSTERING
    # -----------------------------
    st.header("🧠 Clustering")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    k = st.slider("Choose number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(rfm_scaled)
    rfm['PCA1'], rfm['PCA2'] = pca_components[:, 0], pca_components[:, 1]
    st.plotly_chart(px.scatter(rfm, x='PCA1', y='PCA2', color='Cluster', title="PCA Cluster Visualization"))

    # -----------------------------
    # [5] PRODUCT SIMILARITY
    # -----------------------------
    st.header("🔍 Product Similarity")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    df['BERT_Embeddings'] = df['Description'].apply(lambda x: model.encode(x))

    def get_similar_products(desc):
        vector = model.encode([desc])
        distances = [cosine(vector, vec) for vec in df['BERT_Embeddings']]
        df['Similarity'] = distances
        return df.nsmallest(5, 'Similarity')[['StockCode', 'Description']]

    query = st.text_input("Enter a product description")
    if query:
        st.dataframe(get_similar_products(query))

    # -----------------------------
    # [6] CLV PREDICTION
    # -----------------------------
    st.header("💰 CLV Prediction")
    X = rfm[['Recency', 'Frequency', 'AvgOrderValue', 'Tenure']]
    y = rfm['Monetary']
    clv_model = XGBRegressor(n_estimators=200, learning_rate=0.05)
    clv_model.fit(X, y)
    rfm['Predicted CLV'] = clv_model.predict(X)
    st.plotly_chart(px.histogram(rfm, x='Predicted CLV', title="Predicted CLV Distribution"))

    # -----------------------------
    # [7] MODEL SAVING
    # -----------------------------
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(clv_model, 'clv_model.pkl')

    # -----------------------------
    # [8] INSIGHTS & CONCLUSION
    # -----------------------------
    st.header("📌 Insights")
    st.plotly_chart(px.box(rfm, x='Cluster', y='Monetary', color='Cluster'))
    st.dataframe(rfm.groupby('Cluster').mean())
else:
    st.info("Upload a CSV file to get started.")
