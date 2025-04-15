#!/usr/bin/env python
# coding: utf-8

# # E Commerce Customer Segmentation
1. Importing Required LibrariesThese are necessary for data processing, visualization, modeling, clustering, and product similarity analysis.
# In[1]:


#importing required liabraies

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import warnings
warnings.filterwarnings('ignore')


# 2.Data Loading & Preparation:
This section loads the dataset and cleans it by handling missing values and creating new features.
# In[2]:


#load data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df.drop_duplicates(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    return df


# In[3]:


# File path (update before running)
file_path = "E:\E commerce customer\data.csv"
df = load_data(file_path)


# In[4]:


df.head(1)


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# 3.EDA (Exploratory Data Analysis)
This section explores the dataset, providing summaries, visualizations, and checks for missing values.italicized text
# In[7]:


# Check for missing values and data types
col_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})  # Check column types
# Use pd.concat instead of append
col_info = pd.concat([col_info, pd.DataFrame(df.isnull().sum()).T.rename(index={0: 'null values (nb)'})])  # Check missing values
col_info = pd.concat([col_info, pd.DataFrame(df.isnull().sum() / df.shape[0] * 100).T.rename(index={0: 'null values (%)'})])  # Check missing values in percentage
col_info


# In[8]:


# Summary statistics
print(df.describe())


# In[9]:


# Plot the distribution of TotalSales

fig = px.histogram(df, x='TotalSales', nbins=50, title='Distribution of Total Sales',
                   labels={'TotalSales': 'Total Sales'},
                   marginal='box',  # Optional: adds a box plot on top
                   opacity=0.75)

fig.update_layout(
    xaxis_title='Total Sales',
    yaxis_title='Frequency',
    bargap=0.1
)

fig.show()


# In[10]:


# Plot the distribution of Quantity
fig = px.histogram(df, x='Quantity', nbins=50,
                   title='Distribution of Quantity',
                   labels={'Quantity': 'Quantity'},
                   marginal='box',  # Optional: adds a box plot on the side
                   opacity=0.75)
fig.update_layout(
    xaxis_title='Quantity',
    yaxis_title='Frequency',
    bargap=0.1,
    template='plotly_dark'  # or 'plotly_white' based on your preference
)
fig.show()


# In[11]:


#Quantity VS Total Sales
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Quantity', y='TotalSales')
plt.title('Quantity vs Total Sales')
plt.xlabel('Quantity')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# 4.RFM Analysis
In this part, we perform Recency, Frequency, and Monetary (RFM) analysis to segment the customers based on their buying behavior.italicized text
# In[12]:


# RFM Analysis
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSales': 'sum'
})


# In[13]:


rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.dropna(inplace=True)
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
rfm['Tenure'] = (snapshot_date - df.groupby('CustomerID')['InvoiceDate'].min()).dt.days


# 5.Feature Scaling & Clustering
Here, we scale the data for clustering and perform KMeans clustering on the RFM data.italicized text
# In[14]:


# Feature Engineering (Example - Average Basket Size)
df['BasketSize'] = df.groupby('InvoiceNo')['StockCode'].transform('count')
rfm['AvgBasketSize'] = df.groupby('CustomerID')['BasketSize'].mean()


# In[15]:


# Scaling Data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)


# In[16]:


# KMeans Clustering
silhouettes = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    silhouettes.append(silhouette_score(rfm_scaled, kmeans.labels_))
optimal_k = silhouettes.index(max(silhouettes)) + 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


# In[ ]:


from sklearn.metrics import silhouette_score

silhouettes = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    silhouettes.append(silhouette_score(rfm_scaled, kmeans.labels_))

optimal_k = silhouettes.index(max(silhouettes)) + 2  # Best k value
print(f"✅ Optimal number of clusters based on silhouette score: {optimal_k}")

# Fit final KMeans model with optimal_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


# 6. Cluster Visualization (PCA, t-SNE, Heatmap)
Visualizes clustering results in 2D and 3D using PCA and t-SNE.
Also shows heatmap of average RFM values per cluster.
# In[ ]:


# Visualization 1: Cluster Distribution (Heatmap)
plt.figure(figsize=(8,5))
sns.heatmap(rfm.groupby('Cluster').mean(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Cluster Heatmap")
plt.show()


# In[ ]:


# Visualization 2: 3D Clustering with PCA (Principal Component Analysis)
pca = PCA(n_components=3)
pca_components = pca.fit_transform(rfm_scaled)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_components[:, 0], pca_components[:, 1], pca_components[:, 2], c=rfm['Cluster'], cmap='viridis', s=50)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title("3D Cluster Visualization with PCA")
plt.show()


# In[ ]:


# Visualization 3: 2D t-SNE Clustering
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(rfm_scaled)
plt.figure(figsize=(10,7))
sns.scatterplot(x=tsne_components[:, 0], y=tsne_components[:, 1], hue=rfm['Cluster'], palette="viridis", s=100)
plt.title("2D t-SNE Visualization of Clusters")
plt.show()


# 7.Product Similarity Using BERT Embeddings
This section calculates the similarity between products using NLP techniques (BERT embeddings and cosine similarity).italicized text
# In[ ]:


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
df['BERT_Embeddings'] = df['Description'].apply(lambda x: model.encode(x))


# In[ ]:


def get_similar_products(product_desc):
    product_vector = model.encode([product_desc])
    distances = [cosine(product_vector, vec) for vec in df['BERT_Embeddings']]
    df['Similarity'] = distances
    return df.nsmallest(5, 'Similarity')[['StockCode', 'Description']]


# 8.Customer Lifetime Value (CLV) Prediction
Predicts future value of a customer using XGBoost based on RFM features. Useful for targeting high-value customers.
# In[ ]:


X = rfm[['Recency', 'Frequency', 'AvgOrderValue', 'Tenure']]
y = rfm['Monetary']
clv_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
clv_model.fit(X, y)
rfm['Predicted CLV'] = clv_model.predict(X)


# 9.Save Models for Reuse
Saves trained models and scalers so you can use them later without retraining.
# In[ ]:


joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(clv_model, "clv_model.pkl")


# 10.CLV and Cluster Insights Visualization
Visualizes predicted CLV and compares monetary value across different clusters. Great for deriving business insights.
# In[ ]:


plt.figure(figsize=(8,6))
sns.histplot(rfm['Predicted CLV'], kde=True, color='skyblue')
plt.title("Distribution of Predicted Customer Lifetime Value (CLV)")
plt.show()


# In[ ]:


# Insights per Cluster
print(f"Cluster Centers:\n{rfm.groupby('Cluster').mean()}")


# In[ ]:


sns.boxplot(x='Cluster', y='Monetary', data=rfm)
plt.title('Monetary Distribution by Cluster')
plt.show()


# In[ ]:


# Show predictions for top customers
print(f"Predicted CLV for first few customers:\n{rfm[['Predicted CLV']].head().set_index(rfm.head().index.to_series().rename('CustomerID'))}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




