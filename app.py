import streamlit as st
import pandas as pd
import joblib

# 1. PERSIAPAN MUAT ARTEFAK DENGAN CACHE
@st.cache_resource
def load_artefak():
    scaler = joblib.load('scaler_lazada.joblib')
    model_kmeans = joblib.load('kmeans_lazada.joblib')
    return scaler, model_kmeans

scaler, model_kmeans = load_artefak()

# 2. UI STREAMLIT
st.title(" Prediksi Segmentasi / Cluster Produk Lazada")
st.write("Aplikasi dasbor interaktif untuk mengelompokkan produk berdasarkan karakteristiknya.")

st.markdown("---")

# Sesuaikan input dengan fitur yang digunakan saat pelatihan model di notebook
col1, col2 = st.columns(2)

with col1:
    # Contoh jika fitur pertama adalah Price
    price_val = st.number_input("Harga Produk (Price)", min_value=0.0, value=50000.0, step=1000.0)

with col2:
    # Contoh jika fitur kedua adalah Rating / Review Count
    rating_val = st.number_input("Rating Produk", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

# 3. LOGIKA PREDIKSI
if st.button("Prediksi Cluster"):
    # PASTIKAN nama kolom dan urutannya sama persis seperti di notebook saat scaling
    input_data = pd.DataFrame(
        [[price_val, rating_val]], 
        columns=['price', 'rating']  # <-- Ganti dengan nama kolom di dataset Anda
    )
    
    # Standarisasi data input
    scaled_input = scaler.transform(input_data)
    
    # Prediksi cluster
    cluster_result = model_kmeans.predict(scaled_input)[0]
    
    # Tampilkan hasil
    st.success(f"Produk ini masuk ke dalam: **Cluster {cluster_result}**")
    st.metric(label="Status Prediksi", value="Berhasil", delta="Selesai")