import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tcn import TCN

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Kripto Ethereum - TCN", layout="wide")
st.title("Prediksi Harga Kripto Ethereum (USD) menggunakan Model TCN")

# Sidebar tetap tampil apa pun kondisinya
with st.sidebar:
    st.markdown("""
    Unggah file **CSV** yang berisi data harga kripto Ethereum.
    Harga harus dalam satuan **USD**.
    """)
    uploaded_file = st.file_uploader(
        "Upload file CSV Anda",
        type="csv",
        help="Pastikan file memiliki kolom: Date, Close"
    )

# Cek apakah user sudah upload file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.success("Dataset berhasil dimuat.")

    st.subheader("Data 30 Hari Terakhir")
    st.dataframe(df.tail(30), use_container_width=True)

    # Preprocessing
    close_df = df[["Date", "Close"]].copy()
    close_only = close_df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_only)

    # Split data
    training_size = int(len(scaled_close) * 0.80)
    train_data = scaled_close[:training_size]
    test_data = scaled_close[training_size:]

    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step])
            y.append(data[i + time_step])
        return np.array(X), np.array(y)

    time_step = 30
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    try:
        model = load_model("model_prediksi_ethereum MAE75,70.keras", custom_objects={'TCN': TCN})
        st.success("Model berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # Prediksi data uji
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Ambil data terakhir
    actual_last30 = original_ytest[-30:].flatten()
    predicted_last30 = test_predict[-30:].flatten()
    test_dates = df['Date'].iloc[len(train_data) + time_step:]
    dates_last30 = test_dates.reset_index(drop=True).iloc[-30:]

    comparison_df = pd.DataFrame({
        'Tanggal': dates_last30.values,
        'Harga Aktual': actual_last30,
        'Harga Prediksi': predicted_last30
    })

    st.subheader("Perbandingan 30 Hari Terakhir Aktual vs Prediksi (Close Price)")
    st.dataframe(comparison_df.round(2), use_container_width=True)

    st.subheader("Grafik 30 Hari Terakhir Aktual vs Prediksi (Close Price)")
    st.line_chart(comparison_df.set_index("Tanggal"))

    # Prediksi 1 hari ke depan
    x_input = test_data[-time_step:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []

    for _ in range(1):
        x_pred = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        yhat = model.predict(x_pred, verbose=0)
        lst_output.append(yhat[0][0])
        temp_input.append(yhat[0][0])

    future_pred = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = [last_date + datetime.timedelta(days=i + 1) for i in range(1)]

    future_df = pd.DataFrame({
        "Tanggal": future_dates,
        "Harga Prediksi (USD)": future_pred
    })

    st.subheader("Prediksi Harga 1 Hari ke Depan")
    st.dataframe(future_df.round(2), use_container_width=True)

else:
    st.warning("Dataset belum tersedia. Silakan upload file CSV.")
