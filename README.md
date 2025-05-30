# TCN_eth_predict
# Prediksi Harga Kripto Ethereum dengan Model TCN

Aplikasi web sederhana menggunakan Streamlit untuk memprediksi harga cryptocurrency Ethereum (ETH) berdasarkan model Temporal Convolutional Network (TCN) yang sudah dilatih dan disimpan dalam format `.keras`.

---

## Fitur

- Upload dataset Ethereum (ETH) dari file CSV yang terdiri dari kolom date, open, high, low, close, volume.
- Melakukan preprocessing data dan scaling harga.
- Memuat model TCN yang sudah dilatih dari file `.keras`.
- Menampilkan perbandingan harga aktual dan prediksi harga 30 hari terakhir.
- Menampilkan prediksi harga 1 hari ke depan.
  

---

## Cara Menjalankan

1. **1. Siapkan lingkungan Python dan dependencies**

   Pastikan sudah install Python 3.7+ dan dependencies dari `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   
**2. Siapkan file model**

Letakkan file model yang sudah dilatih bernama model_prediksi_ethereum.keras pada direktori yang sama dengan file aplikasi.

**3. Jalankan aplikasi**

Gunakan perintah berikut di terminal:

bash
Copy
Edit
streamlit run app.py

**5. Buka aplikasi di browser**

Biasanya akan terbuka otomatis.

**Struktur File**
- app.py : Script utama aplikasi Streamlit.
- model_prediksi_ethereum.keras : File model TCN hasil pelatihan.
- requirements.txt : Daftar paket Python yang dibutuhkan.

**Library yang Digunakan**
Streamlit

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

**Catatan**
Dataset harus memiliki minimal kolom Date dan Close.

Model yang digunakan harus sudah dilatih sebelumnya dengan arsitektur TCN.
