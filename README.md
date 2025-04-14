#  E-commerce Customer Segmentation Project

This project focuses on analyzing and segmenting e-commerce customers using behavioral metrics like Recency, Frequency, and Monetary value (RFM), clustering techniques, product similarity via BERT embeddings, and Customer Lifetime Value (CLV) prediction using XGBoost. It also supports visual analytics and model deployment via Streamlit.

---

## 📊 Features

- 📊 Exploratory Data Analysis (EDA)
- 📦 RFM Feature Engineering
- 🧠 Customer Segmentation using KMeans
- 📉 Dimensionality Reduction with PCA & t-SNE
- 🔍 Product Recommendation using BERT + Cosine Similarity
- 💰 CLV Prediction using XGBoost
- 🖼️ Visualizations using Plotly & Seaborn
- 🌐 Streamlit Dashboard for interactivity

---

## 🔪 Project Workflow

### 1. **Import Libraries**
Import necessary Python libraries for data manipulation, ML, and visualization.

### 2. **Data Loading and Cleaning**
- Read CSV dataset
- Remove nulls and duplicates
- Filter invalid `Quantity` and `UnitPrice`
- Create `TotalSales = Quantity * UnitPrice`

### 3. **Exploratory Data Analysis (EDA)**
- View summary statistics
- Visualize `TotalSales` and `Quantity` distributions
- Scatter plot for `Quantity vs TotalSales`

### 4. **RFM Analysis**
- Recency: Days since last purchase
- Frequency: Total transactions
- Monetary: Total spend
- Additional features: `AvgOrderValue`, `Tenure`, `AvgBasketSize`

### 5. **Clustering**
- Scale features with `StandardScaler`
- Use `KMeans` and `Silhouette Score` to find optimal `k`
- Visualize clusters using:
  - Cluster Heatmap
  - PCA (3D)
  - t-SNE (2D)

### 6. **Product Similarity**
- Use `Sentence-BERT` (MiniLM) to vectorize product descriptions
- Compute cosine similarity
- Return top 5 similar products for a given input

### 7. **CLV Prediction**
- Features: `Recency`, `Frequency`, `AvgOrderValue`, `Tenure`
- Target: `Monetary`
- Model: `XGBoost Regressor`
- Visualize predicted CLV distribution

### 8. **Model Saving**
- Save models: `Scaler`, `KMeans`, and `CLV Model` using `joblib`

### 9. **Insights and Visualizations**
- Boxplots and mean values per cluster
- Predicted CLV distribution
- Example predictions and segmentation summaries

---

## 📂 Folder Structure

```
📁 ecommerce-segmentation/
│
├── app/                  # Streamlit dashboard
├── models/               # Saved models (KMeans, XGBoost, Scaler)
├── data/                 # Dataset location (not included)
├── notebooks/            # Development notebooks
├── main.py               # Main analysis script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚠️ Dataset Note
The dataset file (approx. 40MB) is too large for GitHub. Please download it from:
- [Kaggle: Online Retail Dataset](https://www.kaggle.com/) or use your own dataset.
- Place it in the `/data` folder and update the file path in the script.

---

## ▶️ Run the Project

### To run analysis:
```bash
python main.py
```

### To launch dashboard:
```bash
streamlit run app/dashboard.py
```

---

## 📋 Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📜 Project Description

This project combines data analytics, machine learning, and natural language processing to segment customers and extract business insights from e-commerce transactional data. It consists of two major components:

### 🔧 1. `main.py` – Core Data Analysis and Modeling Script

The `main.py` script serves as the engine of the project and handles the complete pipeline:

- **Data Preprocessing**: Cleans and processes transactional data by removing invalid entries and calculating new features like `TotalSales`.
- **RFM Analysis**: Computes Recency, Frequency, and Monetary values per customer and engineers additional features such as `Tenure` and `AvgBasketSize`.
- **Clustering**: Applies scaling and KMeans clustering, supported by dimensionality reduction using PCA and t-SNE for visualization.
- **Product Recommendation**: Utilizes Sentence-BERT to encode product descriptions and returns similar products using cosine similarity.
- **CLV Prediction**: Trains an XGBoost model to estimate Customer Lifetime Value (CLV) based on RFM and behavioral metrics.
- **Model Saving**: Exports trained models (`scaler.pkl`, `kmeans_model.pkl`, `clv_model.pkl`) using `joblib` for reuse in the dashboard or other applications.

This script is ideal for data exploration, model building, and offline analysis.

---

### 🌐 2. `app/dashboard.py` – Streamlit Dashboard

The `dashboard.py` file (under the `app/` folder) provides an interactive user interface powered by **Streamlit**, allowing users to:

- Visualize key metrics and EDA graphs
- Interactively explore customer segments
- Input a product description to receive **similar product recommendations**
- Predict Customer Lifetime Value (CLV) for individual customers
- View cluster-wise statistics and business insights

This component is designed for stakeholders, analysts, or business users to explore the insights without touching code.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing
Pull requests and suggestions are welcome! Please open an issue to discuss changes.

---

## 👨‍💼 Author
Made with M.Gowri Shankar
- 📧 Email: gowrishawn123@gmail.com
- 🔗 LinkedIn: https://www.linkedin.com/in/gowri-shankar-m-0544a8137/


