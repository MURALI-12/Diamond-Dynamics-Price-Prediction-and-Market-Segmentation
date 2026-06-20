import streamlit as st
import numpy as np
import pickle

reg_model = pickle.load(open('RandomForest.pkl', 'rb')) 
clust_model = pickle.load(open('best_clustering_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
scaler_clust = pickle.load(open('scaler_clust.pkl', 'rb'))

# Cluster name mapping
cluster_names = {
    0: "Premium Heavy Diamonds",
    1: "Affordable Small Diamonds",
    2: "Mid-range Balanced Diamonds"
}

st.title("💎 Diamond Dynamics")
st.subheader("Predict Diamond Price or Market Segment")

tab1, tab2 = st.tabs(["📈 Price Prediction", "📊 Cluster Prediction"])

with st.sidebar:
    st.header("🔧 Input Features")
    carat = st.number_input("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.0)
    x = st.number_input("Length (x)", min_value=0.0, max_value=10.0, value=0.0)
    y = st.number_input("Width (y)", min_value=0.0, max_value=10.0, value=0.0)
    z = st.number_input("Depth (z)", min_value=0.0, max_value=10.0, value=0.0)

    cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Feature engineering
depth = 61.0  
table = 55.0  
volume = x * y * z
cut_ord = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'].index(cut)
color_ord = ['J', 'I', 'H', 'G', 'F', 'E', 'D'].index(color)
clarity_ord = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'].index(clarity)

features = [carat, x, y, z, depth, table, volume, cut_ord, color_ord, clarity_ord]
features_array = np.array(features).reshape(1, -1)

# Tab 1: Price Prediction
with tab1:
    st.markdown("### 💰 Predict the Price of a Diamond")
    if st.button("Predict Price"):
        scaled_input = scaler.transform(features_array)
        price_pred = reg_model.predict(scaled_input)[0]
        st.success(f"💎 Estimated Price: ₹{price_pred:,.0f}")

# Tab 2: Cluster Prediction
with tab2:
    st.markdown("### 🧠 Predict the Market Segment")
    if st.button("Predict Cluster"):
        scaled_clust_input = scaler_clust.transform(features_array)
        cluster_id = clust_model.predict(scaled_clust_input)[0]
        cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        st.info(f"📊 Segment: {cluster_label} (Cluster {cluster_id})")
