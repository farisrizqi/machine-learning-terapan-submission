# Prediksi Curah Hujan Harian di Jakarta Menggunakan LSTM

Author: Faris Nur Rizqiawan <br>
Platform: Dicoding Submission – Proyek Pertama Machine Learning Terapan <br>
Domain: Lingkungan <br>
Metode: Deep Learning LSTM dan Klasifikasi Biner

## 1. Domain Proyek

### Latar Belakang

Curah hujan adalah salah satu parameter cuaca yang paling krusial untuk berbagai sektor, mulai dari pertanian, pengelolaan air, transportasi, hingga mitigasi bencana. Prediksi curah hujan yang akurat dapat membantu petani dalam merencanakan masa tanam, pemerintah dalam mengantisipasi banjir, serta masyarakat umum dalam aktivitas sehari-hari.

Dengan meningkatnya intensitas perubahan iklim, prediksi berbasis data historis menjadi salah satu pendekatan penting yang didukung oleh kemajuan teknologi dan machine learning. Oleh karena itu, dalam proyek ini akan dibangun sistem prediksi curah hujan selama 14 hari ke depan berdasarkan data historis cuaca di Jakarta.

### Mengapa Masalah Ini Perlu Diselesaikan?
Prediksi curah hujan yang akurat memiliki dampak signifikan bagi berbagai sektor, meliputi pertanian, manajemen bencana, dan perencanaan infrastruktur kota. 

* Membantu petani merencanakan masa tanam dan panen.
* Mempermudah pemerintah dan BPBD dalam antisipasi serta kesiapsiagaan dalam menghadapi bencana banjir atau kekeringan.
* Membantu perencanaan infrastruktur drainase dan sistem peringatan dini bencana.

### Referensi:

* Data curah hujan seringkali menjadi fokus penelitian prediksi cuaca untuk berbagai aplikasi [[sumber_1]](https://jurnal-id.com/index.php/jupin/article/download/99/84).
* Model seperti LSTM dan Backpropagation umum digunakan dalam penelitian prediksi curah hujan [[sumber_2]](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/12022).

## 2. Business Understanding

### Problem Statement

Bagaimana cara memprediksi curah hujan harian selama 14 hari ke depan berdasarkan data historis cuaca Jakarta?

### Goals

Membangun model prediksi curah hujan menggunakan pendekatan time series forecasting untuk memberikan estimasi curah hujan harian 14 hari ke depan dengan akurasi yang dapat diandalkan.

### Solution Statement

Untuk mencapai tujuan di atas, berikut dua pendekatan solusi yang akan dibandingkan:
1. Model LSTM (Long Short-Term Memory)
   * LSTM merupakan arsitektur Recurrent Neural Network (RNN) yang sangat cocok untuk data sekuensial dan time series.
   * Diharapkan mampu menangkap pola temporal jangka panjang dalam data cuaca historis.

2. Baseline Model: Multilayer Perceptron (MLP)
   * Digunakan sebagai pembanding awal untuk mengukur sejauh mana model sekuensial seperti LSTM memberikan peningkatan performa.
   *  Model ini tidak mempertimbangkan dependensi temporal.

## 3. Data Understanding

### Dataset
* Nama: `WeatherJakarta2013-2020.csv`
* Link: [WeatherJakarta2013-2020](https://www.kaggle.com/datasets/farisrizqiawan/weatherjakartaselectedcoloumn2013-2020)

### Deskripsi Dataset
Dataset ini merupakan data historis cuaca harian wilayah DKI Jakarta dari tahun 2013 hingga 2020 (total 2922 data harian). Dataset ini cocok untuk pendekatan time series forecasting, terutama untuk memprediksi curah hujan berdasarkan pola historis cuaca.

### Statistik dataset
- Jumlah total data: 2922 baris (data harian)
- Tidak ada nilai null pada semua kolom
- Rentang waktu: 1 Januari 2013 hingga 31 Desember 2020
- Distribusi nilai curah hujan:
  - Rata-rata: ~5.7 mm/hari
  - Maksimum: ~99.5 mm
  - Mayoritas data memiliki nilai curah hujan rendah (0–10 mm)

### Fitur Dataset antara lain:

| Fitur                             | Deskripsi                                                    |
|-----------------------------------|--------------------------------------------------------------|
| time                              | Waktu atau tanggal pencatatan data.                          |
| weathercode (wmo code)            | Kode cuaca berdasarkan standar WMO (World Meteorological Organization). |
| temperature_2m_max (°C)           | Suhu maksimum di ketinggian 2 meter dalam Celcius.            |
| temperature_2m_min (°C)           | Suhu minimum di ketinggian 2 meter dalam Celcius.            |
| temperature_2m_mean (°C)          | Suhu rata-rata di ketinggian 2 meter dalam Celcius.            |
| apparent_temperature_max (°C)     | Suhu maksimum yang dirasakan (apparent temperature) dalam Celcius. |
| apparent_temperature_min (°C)     | Suhu minimum yang dirasakan (apparent temperature) dalam Celcius. |
| apparent_temperature_mean (°C)    | Suhu rata-rata yang dirasakan (apparent temperature) dalam Celcius. |
| precipitation_sum (mm)            | Total presipitasi (curah hujan, salju, dll) dalam milimeter.  |
| rain_sum (mm)                     | Total curah hujan dalam milimeter.                            |
| precipitation_hours (h)           | Durasi presipitasi (curah hujan, dll) dalam jam.              |
| windspeed_10m_max (km/h)          | Kecepatan angin maksimum di ketinggian 10 meter dalam kilometer per jam. |

### Exploratory Data Analysis (EDA)

Beberapa langkah yang dilakukan:

* Visualisasi tren tahunan dan musiman curah hujan
* Statistik deskriptif untuk curah hujan
* Korelasi antara fitur-fitur numerik terhadap target (`Curah_Hujan`)

## 4. Data Preparation

### Pembersihan Data
- Tanggal: Data telah diurutkan berdasarkan kolom time sebagai time index.
- Duplikasi: Data diperiksa dari kemungkinan duplikasi, dan jika ditemukan akan dihapus.
- Missing Value: Tidak ditemukan missing value dalam dataset.
- Outlier: Nilai ekstrim pada kolom precipitation_sum (mm) telah diperiksa secara statistik. Sebagian besar nilai berada dalam rentang wajar, tetapi tetap disarankan untuk menerapkan transformasi (seperti log) jika distribusi sangat skewed.

### Processing fitur
- Seleksi Fitur:
    ```
    features = [
        'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'temperature_2m_mean (°C)',
        'apparent_temperature_max (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)',
        'precipitation_sum (mm)', 'rain_sum (mm)', 'precipitation_hours (h)', 'windspeed_10m_max (km/h)'
    ]
    ```
- Standarisasi data fitur menggunakan `StandardScaler()` dari library `sklearn.preprocessing` yang di mana setiap fitur diubah agar memiliki rata-rata 0 dan standar deviasi 1.
- Pembuatan window data untuk input LSTM dengan ukuran `WINDOW_SIZE = 14`.
- Split data menjadi data latih dan uji dengan rasio 80:20 secara time-based.


## 5. Modeling

### Pendekatan Model
Karena permasalahan yang dihadapi adalah time series forecasting untuk curah hujan harian, maka model yang dipilih adalah LSTM (Long Short-Term Memory) — salah satu arsitektur Recurrent Neural Network (RNN) yang cocok untuk memproses data sekuensial atau temporal. Berikut adalah gambar alur logika dari arsitektur model LSTM.

<p align="center">
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AoJcSMhQZA3vr0P-kXz0SYA.png" alt="Arsitektur Model LSTM" width="500"/>
</p>

### Penjelasan Model LSTM
LSTM (Long Short-Term Memory) adalah jenis Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient. Arsitektur LSTM terdiri dari beberapa komponen utama:
- **Forget Gate**: Menentukan informasi apa yang harus dilupakan dari cell state sebelumnya.
- **Input Gate**: Memilih informasi baru yang akan ditambahkan ke cell state.
- **Cell State**: Membawa informasi dari waktu sebelumnya, dimodifikasi oleh forget dan input gate.
- **Output Gate**: Menentukan informasi yang akan dikirim ke hidden state berikutnya.

### Arsitektur Model:
1. LSTM Layer
    - Jumlah unit: 64
    - Input shape: (WINDOW_SIZE, jumlah fitur)
    - return_sequences: False
      <br>Layer ini bertugas memproses urutan data cuaca harian dan mengekstraksi pola dari sekuens tersebut.

2. Dropout Layer
    - Dropout rate: 0.3
      <br>Dropout digunakan untuk mencegah overfitting dengan mengabaikan secara acak beberapa neuron selama pelatihan.

3. Dense Hidden Layer
    - Jumlah unit: 32
    - Fungsi aktivasi: ReLU (activation='relu')
      <br>Layer ini berfungsi untuk menangkap hubungan non-linear sebelum masuk ke output.

4. Dense Output Layer
    - Jumlah unit: 1
    - Fungsi aktivasi: Sigmoid (activation='sigmoid')
      <br>Layer ini menghasilkan output probabilitas untuk klasifikasi biner: hujan atau tidak hujan.

### Parameter Pelatihan Model:
- Loss function: `binary_crossentropy` (karena target klasifikasi hujan/tidak hujan)
- Optimizer: `Adam`, learning rate = 0.001
- Epochs: 30
- Batch size: 32
- Validation split: 0.2 (20% dari data latih digunakan sebagai data validasi)
- EarlyStopping:
    - Monitor: val_loss
    - Patience: 5
    - restore_best_weights=True untuk mengambil bobot model terbaik selama pelatihan

### Train Model
Model dilatih selama beberapa epoch dengan pembagian data:
- Training set: 80%
- Validation set: 20%

## 6. Evaluation

### Metrik Evaluasi:
Model dievaluasi menggunakan confusion matrix, classification report, dan grafik akurasi serta loss untuk mengukur performa pada data latih dan uji. 

Matriks Evaluasi:
- Accuracy — proporsi prediksi yang benar dari total prediksi
- Precision — seberapa tepat prediksi "hujan" yang diberikan model
- Recall — seberapa banyak kejadian hujan yang berhasil dideteksi model
- F1-Score — harmonic mean antara Precision dan Recall

Model dievaluasi menggunakan metrik klasifikasi untuk dua kelas: **"Tidak Hujan"** dan **"Hujan"**. Berikut hasil lengkapnya:

| Kelas         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Tidak Hujan   | 0.68      | 0.63   | 0.65     | 159     |
| Hujan         | 0.86      | 0.89   | 0.88     | 423     |
| **Accuracy**  |           |        | **0.82** | **582** |
| Macro Avg     | 0.77      | 0.76   | 0.76     | 582     |
| Weighted Avg  | 0.81      | 0.82   | 0.81     | 582     |


Selain itu, dilakukan prediksi terhadap data uji untuk membandingkan hasil prediksi dengan nilai aktual. 

Hasil menunjukkan bahwa model memiliki akurasi yang cukup baik dan loss yang rendah, menandakan bahwa model mampu belajar dengan efektif dari data historis. Prediksi model juga cukup akurat dalam membedakan hari hujan dan tidak hujan.

## 7. Kesimpulan

Model LSTM berhasil dibangun dan digunakan untuk memprediksi kemungkinan terjadinya hujan pada hari ke-15 berdasarkan pola cuaca 14 hari sebelumnya, menggunakan data historis cuaca Jakarta tahun 2013–2020.

Beberapa poin penting dari proyek ini:

- Model mampu mencapai **akurasi sekitar 82%** pada data uji, menunjukkan performa yang cukup baik dalam klasifikasi biner (hujan / tidak hujan).
- Evaluasi metrik menunjukkan model sangat baik dalam mengenali kelas **"Hujan"** (F1-Score: 0.88), meskipun performa pada kelas **"Tidak Hujan"** masih bisa ditingkatkan.
- Grafik akurasi dan loss selama pelatihan menunjukkan proses pembelajaran berjalan stabil tanpa indikasi overfitting.
- Model mampu menangkap pola temporal dari data cuaca harian dan dapat dijadikan dasar pengembangan sistem prediksi cuaca sederhana.
- Ke depannya, model ini dapat ditingkatkan dengan:
  - Penambahan fitur cuaca lain (misalnya kelembapan, tekanan udara),
  - Penyesuaian arsitektur dan hyperparameter (seperti jumlah unit, dropout, atau window size),
  - Penggunaan model sekuensial lain seperti GRU atau Transformer untuk perbandingan performa.


Model ini dapat digunakan sebagai dasar sistem prediksi cuaca sederhana dan bisa dikembangkan lebih lanjut dengan data tambahan atau tuning hyperparameter lebih lanjut.

