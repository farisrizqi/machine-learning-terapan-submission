# Prediksi Curah Hujan Harian di Jakarta Menggunakan LSTM

## 1. Domain Proyek

### Latar Belakang

Curah hujan adalah salah satu faktor krusial dalam sistem cuaca yang berdampak besar pada sektor pertanian, pengelolaan air, transportasi, dan mitigasi bencana. Akurasi prediksi curah hujan sangat penting untuk perencanaan dan pengambilan keputusan di berbagai bidang.

Dalam konteks ini, dibangunlah sistem prediksi curah hujan harian selama 14 hari ke depan berdasarkan data historis cuaca di Jakarta. Model Long Short-Term Memory (LSTM) dipilih karena kemampuannya dalam mempelajari pola jangka panjang dari data time series. Selain itu, sebagai pembanding, juga digunakan model Multilayer Perceptron (MLP) sebagai baseline.

### Mengapa Masalah Ini Perlu Diselesaikan?

* Membantu petani merencanakan masa tanam dan panen.
* Mempermudah pemerintah dan BPBD dalam antisipasi banjir atau kekeringan.
* Membantu perencanaan infrastruktur drainase dan sistem peringatan dini bencana.

### Referensi:

* [Jurnal Prediksi Curah Hujan](https://jurnal-id.com/index.php/jupin/article/download/99/84)
* [Studi LSTM untuk Prediksi Cuaca](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/12022)

## 2. Business Understanding

### Problem Statement

Bagaimana cara memprediksi curah hujan harian selama 14 hari ke depan berdasarkan data historis cuaca di Jakarta?

### Goals

Mengembangkan model machine learning berbasis time series forecasting untuk memprediksi curah hujan harian selama 14 hari ke depan dengan akurasi yang dapat diandalkan.

### Solution Statement

Terdapat dua pendekatan solusi yang dieksplorasi:

1. **LSTM**: Model deep learning untuk data time series, mampu menangkap dependensi temporal.
2. **MLP**: Baseline model untuk evaluasi performa dibandingkan model sekuensial.

## 3. Data Understanding

### Dataset

* Nama: `WeatherJakarta2013-2020.csv`
* Jumlah sampel: >2500 baris (8 tahun data harian)
* Fitur yang tersedia antara lain:

  * `Tanggal`
  * `Curah_Hujan`
  * `Suhu_Rata_Rata`
  * `Kelembaban_Relatif`
  * `Kecepatan_Angin`

### Link Dataset:

(Dataset ini merupakan data internal dan tidak berasal dari sumber publik.)

### Exploratory Data Analysis (EDA)

Beberapa langkah yang dilakukan:

* Visualisasi tren tahunan dan musiman curah hujan
* Statistik deskriptif untuk curah hujan
* Korelasi antara fitur-fitur numerik terhadap target (`Curah_Hujan`)

## 4. Data Preparation

### Langkah-langkah:

1. **Konversi Tanggal** menjadi format datetime dan set sebagai index.
2. **Normalisasi** nilai menggunakan MinMaxScaler.
3. **Sliding Window**: Menyusun data menjadi sequence (misal 30 hari terakhir untuk prediksi hari ke-31).
4. **Split Data**: Data dibagi menjadi data latih dan data uji dengan rasio 80:20 secara time-based.

### Alasan Tahapan:

* LSTM dan MLP membutuhkan input dengan format numerik dan skala seragam.
* Pendekatan sliding window diperlukan untuk menangkap pola sekuensial.

## 5. Modeling

### Model 1: LSTM

* Layer: 2 LSTM layer + Dense layer
* Sequence length: 30 hari
* Output: Prediksi curah hujan untuk 1 hari ke depan, dilakukan iteratif selama 14 hari.
* Epochs: 50
* Optimizer: Adam

### Model 2: MLP

* Baseline model
* Fully connected layers dengan input fitur yang sama (flattened window)

### Hyperparameter Tuning

* LSTM: jumlah unit LSTM, ukuran window, batch size
* Metrik evaluasi digunakan untuk memilih model terbaik

### Kelebihan & Kekurangan

| Model | Kelebihan                           | Kekurangan                           |
| ----- | ----------------------------------- | ------------------------------------ |
| LSTM  | Mampu mengenali pola jangka panjang | Waktu latih lebih lama               |
| MLP   | Lebih cepat dan sederhana           | Kurang akurat untuk data time series |

## 6. Evaluation

### Metrik Evaluasi:

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**

### Hasil:

* **LSTM**:

  * RMSE: \~3.1 mm
  * MSE: \~9.6
  * MAE: \~2.4
* **MLP**:

  * RMSE: \~5.2 mm
  * MSE: \~27.0
  * MAE: \~4.5

### Analisis:

Model LSTM menghasilkan error yang lebih rendah secara signifikan dibandingkan MLP, menunjukkan efektivitas dalam mempelajari pola jangka panjang dari curah hujan harian.

## 7. Kesimpulan

* Model LSTM terbukti lebih akurat dibanding baseline MLP dalam memprediksi curah hujan harian di Jakarta.
* Akurasi model cukup untuk digunakan sebagai sistem pendukung keputusan dalam sektor pertanian dan mitigasi bencana.
* Untuk pengembangan lebih lanjut, dapat digunakan data spasial dari daerah lain, atau data atmosfer tambahan seperti tekanan udara dan suhu permukaan laut.

## 8. Referensi

* Jurnal: [https://jurnal-id.com/index.php/jupin/article/download/99/84](https://jurnal-id.com/index.php/jupin/article/download/99/84)
* Studi: [https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/12022](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/12022)
* Dataset internal: WeatherJakarta2013-2020.csv
