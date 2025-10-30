# ==========================================================
# app.py (FINAL VERSION - LENGKAP)
# ==========================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import os
import model_utils as mu
from datetime import date, timedelta
import joblib

# --- KONFIGURASI STREAMLIT ---
st.set_page_config(
    page_title="Sistem Prediksi NO2",
    layout="wide" 
)

# --- UTILITY: Streamlit Caching ---
@st.cache_resource
def load_or_train_model(file_path, n_lags, test_size):
    """Memuat atau melatih model Random Forest dan data pendukung."""
    if not os.path.exists('model'):
        os.makedirs('model')
    
    try:
        # Latih ulang untuk memastikan fitur terbaru
        results = mu.prepare_and_train_all(file_path, n_lags, test_size) 
        return results['rf_model'], results['full_df'], results['last_data'], results['metrics_rf']

    except Exception as e:
        # Menampilkan pesan error yang lebih spesifik
        st.error(f"Sistem tidak dapat berjalan. Pastikan file '{file_path}' ada dan formatnya benar. Error: {e}")
        return None, None, None, None

# ==========================================================
# 0. UI UTAMA & INIT
# ==========================================================
st.title("🏭 Sistem Prediksi NO2 Harian")
st.caption("Aplikasi ini menggunakan model **Random Forest** untuk memprediksi konsentrasi NO₂ harian.")
st.markdown("---")

# Konfigurasi Awal
DATA_FILE = "NO2_Pademawu.csv"
N_LAGS = mu.N_LAGS
TEST_SIZE = mu.TEST_SIZE_DAYS

# --- Memuat/Melatih Model di Badan Utama ---
with st.spinner("Memuat dan melatih model..."):
    rf_model, full_data, last_data, metrics = load_or_train_model(DATA_FILE, N_LAGS, TEST_SIZE)

if rf_model is None:
    st.error("Model Gagal Dimuat/Dilatih. Aplikasi dihentikan.")
    st.stop()

# Tampilkan status keberhasilan
st.success(f"Model Random Forest siap digunakan. Data historis terakhir: {last_data.index[-1].date()}")

# Ambil tanggal terakhir historis
last_historical_date = last_data.index[-1].date()


# ==========================================================
# 1. METRIK KINERJA MODEL
# ==========================================================
st.subheader("✅ Kinerja Model pada Data Uji")
st.markdown("Hasil evaluasi model Random Forest pada data uji terakhir:")
col1, col2 = st.columns(2) 

with col1:
    st.metric(
        label="MAPE",
        value=f"{metrics['mape']:.2f} %",
        help="Mean Absolute Percentage Error. Persentase rata-rata kesalahan prediksi."
    )
with col2:
    st.metric(
        label="ACF Residuals (Lag 1)",
        value=f"{metrics['acf']:.4f}",
        help="Autokorelasi Residual pada lag 1. Nilai mendekati nol menunjukkan autokorelasi rendah (model baik menangkap pola)."
    )

st.markdown("---")

# ==========================================================
# 2. INPUT PERIODE PREDIKSI
# ==========================================================
st.header("🎯 Tentukan Periode Prediksi")

input_container = st.container()

with input_container:
    col_start, col_end = st.columns([1, 2])

    # Tanggal Awal Otomatis
    start_date_forecast = last_historical_date + timedelta(days=1)
    
    with col_start:
        st.date_input(
            "Tanggal Mulai Prediksi (Otomatis)", 
            value=start_date_forecast, 
            disabled=True,
            key="start_date_fixed" 
        )
        # Menampilkan info historis terakhir di sini
        st.markdown(f"**Data Historis Terakhir:** `{last_historical_date}`") 

    # Tanggal Akhir
    with col_end:
        target_date = st.date_input(
            "Pilih Tanggal Akhir Prediksi", 
            min_value=start_date_forecast,
            value=start_date_forecast + timedelta(days=7),
            max_value=start_date_forecast + timedelta(days=60),
            help="Pilih tanggal di masa depan, maksimal 60 hari dari data historis terakhir."
        )

days_to_forecast = (target_date - last_historical_date).days

# ==========================================================
# 3. TOMBOL & PROSES PREDIKSI
# ==========================================================
st.markdown("---")

if st.button(f"🚀 Mulai Prediksi NO₂ untuk {days_to_forecast} Hari (Hingga {target_date})", type="primary"):
    
    if days_to_forecast < 1:
        st.error("Jumlah hari prediksi tidak valid. Pilih tanggal akhir yang lebih jauh dari tanggal historis terakhir.")
        st.stop()
        
    with st.spinner(f"Memprediksi NO2 untuk **{days_to_forecast} hari** menggunakan Random Forest..."):
        
        # --- Prediksi Random Forest ---
        forecast_df = mu.predict_rf_n_days(
            rf_model, last_data, days_to_forecast, N_LAGS
        )

        # Pastikan index datetime dan urutan benar
        forecast_df.index = pd.to_datetime(forecast_df.index)
        forecast_df = forecast_df.sort_index()

    st.success("✅ Prediksi Selesai! Lihat hasilnya di bawah.")

    # Data historis terakhir (90 hari)
    historic_data_plot = full_data['NO2'].tail(90)

    # Format dataframe untuk ditampilkan
    display_df = forecast_df[['NO2_Prediction']].copy()
    display_df.index.name = 'Tanggal'
    display_df.columns = ['Prediksi NO2 (µg/m³)']
    
    # Tambahkan kolom level kualitas udara (Contoh Sederhana)
    def quality_level(no2):
        if no2 < 40: return "Baik (Good)"
        elif no2 < 80: return "Sedang (Moderate)"
        else: return "Tidak Sehat (Unhealthy)"
    
    display_df['Kualitas Udara'] = display_df['Prediksi NO2 (µg/m³)'].apply(quality_level)
    
    
    # ==========================================================
    # 4. HASIL PREDIKSI DENGAN TABS
    # ==========================================================
    st.header("📈 Hasil Prediksi")

    # Membuat Tabs
    tab_graph, tab_table, tab_download = st.tabs(["Grafik Prediksi", "Tabel Detail", "Unduh Hasil"])
    
    with tab_graph:
        st.subheader(f"Grafik Prediksi NO2 ({days_to_forecast} Hari)")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Data Historis
        ax.plot(historic_data_plot.index, historic_data_plot.values, 
                label='Historis (90 Hari Terakhir)', color='#1f77b4', linewidth=2, alpha=0.7)

        # Garis batas awal prediksi
        ax.axvline(x=forecast_df.index.min(), color='black', linestyle='--', linewidth=1, label='Awal Prediksi')

        # Plot hasil prediksi
        ax.plot(forecast_df.index, forecast_df['NO2_Prediction'], 
                label='Prediksi NO2', color='red', linestyle='-', linewidth=2)

        ax.set_title(f"Prediksi NO2 Hingga {target_date}", fontsize=14)
        ax.set_xlabel("Tanggal", fontsize=12)
        ax.set_ylabel("Konsentrasi NO2 (µg/m³)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize='small')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig)

    with tab_table:
        st.subheader("Detail Prediksi Harian")
        st.dataframe(display_df, use_container_width=True)

    with tab_download:
        st.subheader("Unduh Data")
        csv = display_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi Lengkap (.csv)",
            data=csv,
            file_name=f'NO2_Prediksi_RF_Hingga_{target_date}.csv',
            mime='text/csv',
        )
