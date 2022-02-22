# Laporan Proyek Machine Learning - Naufal Aldy Pradana

## Domain Proyek
Jantung merupakan salah satu organ vital pada tubuh manusia yang terletak pada bagian tengah dada, tepatnya di sisi kiri tubuh.  Jantung memiliki fungsi utama untuk memompa darah yang berisi nutrisi dan oksigen ke seluruh tubuh dan mengangkut zat-zat sisa yang tidak lagi dibutuhkan oleh tubuh.  Sehingga apabila jantung dan pembuluhnya bermasalah tentunya akan menimpulkan suatu dampak negatif seperti timbulnya berbagai penyakit jantung seperti serangan jantung dan apabila tidak mendapatkan penanganan yang baik dapat merenggut nyawa sang penderita.  

Serangan jantung terjadi karena terhambatnya aliran darah ke otot jantung dimana penyebab utama kondisi tersebut adalah penyakit jantung koroner yang timbil akibat timbunan kolestrol yang membentuk plat di dinding pembuluh darah ssehingga menyumbat pembuluh darah yang memasok darah ke jantung(pembuluh darah koroner).  Gejala serangan jantung diantaranya yaitu nyeri dada, pusing, sesak napas, dan keringat dingin namun ada juga penderita serangan jantung yang tak mengalami gejala dan fungsi jantung langsung berhenti begitu saja.

Dilansir dari [WHO](https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)), serangan jantung termasuk dari Cardiovasicular Diseases(CVDs) atau penyakit jantung yang banyak merenggut korban jiwa secara global dengan estimasi 17,9 juta jiwa orang meninggal disebabkan oleh CVDs pada tahun 2019, mewakili 32% dari jumlah kematian global di tahun tersebut, 85% diantaranya karena serangan jantung dan stroke.

Sehingga berdasarkan uraian latar belakang tersebut dibuatlah pengerjaan proyek machine learning ini dengan cakupan domain untuk memprediksi kemungkinan serangan jantung berdasarkan hasil ujian pasien.

## Business Understanding
Dalam rangka mengatasi permasalahan serangan jantung yang merenggut banyak kematian secara global seorang ilmuwan data ingin membuat sebuah sistem yang dapat memprediksi kemungkinan seseorang terkena serangan jantung sehingga penderita penyakit jantung bisa mendapatkan penanganan, Ia lalu bekerja sama dengan rumah sakit untuk mensupplynya dengan data-data berkaitan dengan penyebab-penyebab umum yang biasa mempengaruhi serangan jantung.  Rumah sakit pun mengirimkan data tersebut dalam bentuk fitur-fitur yang kemudian harus dinalisa lebih lanjut oleh sang ilmuwan.  Ada sekitar 13 fitur yang diberikan dan semuanya telah diubah kedalam bentuk numerikal sehingga memudahkan proses analisa.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap kemungkinan pasien terkena serangan jantung?
- Bagaimana cara mengetahui kemungkinan pasien lebih mungkin terkena serangan jantung atau tidak berdasarkan fitur-fitur tertentu?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur yang paling berpengaruh terhapdap kemungkinan pasien terkena serangan jantung
- Membuat model machine learning dengan membandingkan beberapa algoritma hingga menjadi model yang seakurat mungkin

## Data Understanding
