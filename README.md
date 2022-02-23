# Laporan Proyek Machine Learning - Naufal Aldy Pradana

## Domain Proyek
Jantung merupakan salah satu organ vital pada tubuh manusia yang terletak pada bagian tengah dada, tepatnya di sisi kiri tubuh.  Jantung memiliki fungsi utama untuk memompa darah yang berisi nutrisi dan oksigen ke seluruh tubuh dan mengangkut zat-zat sisa yang tidak lagi dibutuhkan oleh tubuh.  Sehingga apabila jantung dan pembuluhnya bermasalah tentunya akan menimpulkan suatu dampak negatif seperti timbulnya berbagai penyakit jantung seperti serangan jantung dan apabila tidak mendapatkan penanganan yang baik dapat merenggut nyawa sang penderita.  

Serangan jantung terjadi karena terhambatnya aliran darah ke otot jantung dimana penyebab utama kondisi tersebut adalah penyakit jantung koroner yang timbil akibat timbunan kolestrol yang membentuk plat di dinding pembuluh darah ssehingga menyumbat pembuluh darah yang memasok darah ke jantung(pembuluh darah koroner).  Gejala serangan jantung diantaranya yaitu nyeri dada, pusing, sesak napas, dan keringat dingin namun ada juga penderita serangan jantung yang tak mengalami gejala dan fungsi jantung langsung berhenti begitu saja.

Dilansir dari [WHO](https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)), serangan jantung termasuk dari Cardiovasicular Diseases(CVDs) atau penyakit jantung yang banyak merenggut korban jiwa secara global dengan estimasi 17,9 juta jiwa orang meninggal disebabkan oleh CVDs pada tahun 2019, mewakili 32% dari jumlah kematian global di tahun tersebut, 85% diantaranya karena serangan jantung dan stroke.

Sehingga berdasarkan uraian latar belakang tersebut dibuatlah pengerjaan proyek machine learning ini dengan cakupan domain untuk memprediksi kemungkinan serangan jantung berdasarkan hasil ujian pasien.

## Business Understanding
Dalam rangka mengatasi permasalahan serangan jantung yang merenggut banyak kematian secara global seorang ilmuwan data ingin membuat sebuah sistem yang dapat memprediksi kemungkinan seseorang terkena serangan jantung sehingga penderita penyakit jantung bisa mendapatkan penanganan, Ia lalu bekerja sama dengan rumah sakit untuk mensupplynya dengan data-data berkaitan dengan penyebab-penyebab umum yang biasa mempengaruhi serangan jantung.  Rumah sakit pun mengirimkan data tersebut dalam bentuk fitur-fitur yang kemudian harus dinalisa lebih lanjut oleh sang ilmuwan, dimana ada total 11 fitur yang kemudian harus diproses lebih lanjut.

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, sang ilmuwan tersebut menyimpulkan beberap rumusan masalah sebagai berikut:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap kemungkinan pasien terkena serangan jantung?
- Bagaimana cara mengetahui kemungkinan pasien lebih mungkin terkena serangan jantung atau tidak berdasarkan fitur-fitur tertentu?

### Goals

Untuk menjawab pertanyaan tersebut, sang ilmuwan tersebut membuat model klasifikasi dengan goals sebagai berikut:
- Mengetahui fitur yang paling berpengaruh terhapdap kemungkinan pasien terkena serangan jantung
- Membuat model machine learning yang dapat mengklasifikasikan apakah fitur-fitur yang dimiliki suatu pasien cenderung lebih mungkin terkena serangan jatung atau tidak

### Solution Statement
Untuk mencapai goals yang telah ditetapkan sebelumnya, sang ilmuwan merumuskan solusi sebagai berikut:
- Melakukan explarotary data analysis untuk mengamati keterkaitan antara fitur dengan label yang telah ditetapkan, dan melakukan validasi dengan melihat hasil evaluasi dari tiap fitur penting dari masing-masing model
- Melakukan perbandingan antara beberapa algoritma serta hyperparameter tuning jika base model masih kurang baik performanya

## Data Understanding
Data yang digunakan pada proyek ini adalah dataset Heart Attack Analysis yang dapat diunduh dari [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction).  Menurut authornya dataset ini merupakan dataset analisis jantung yang terbesar karena telah mengkombinasikan dari beberapa dataset jantung lain yang telah ada sebelumnya seperti sehingga memiliki jumlah data final sebanyak 918 data unik.

### Variabel-variabel pada dataset Heart Attack Analysis adalah sebagai berikut:
- Age: merupakan umur pasien dalam tahun
- Sex: merupakan gender dari pasien [M: Male, F: Female]
- ChestPainType: merupakan tipe-tipe sakit pada dada pasien [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: merupakan tekanan darah pasien saat kondisi resting [mm Hg]
- Cholesterol: merupakan jumlah serum kolesterol [mm/dl]
- FastingBS: merupakan hasil tes darah setelah puasa sepanjang malam hingga dites [1: jika hasil > 120 mg/dl, 0: jika sebaliknya]
- RestingECG: merupakan hasil dari tes electrocardiogram  [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: merupakan detak jantung maksimal yang pernah dicapai [60~202]
- ExerciseAngina: merupakan latihan yang dipengaruhi angina(gangguan/rasa tidak enak pada dada) [Y: Yes, N: No]
- Oldpeak: merupakan nilai numerik untuk mengukur depresi
- ST_Slope: merupakan kemiringan saat menjalan segmen latihan ST [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: merupakan label kategori biner yang mengeluarkan 1: apabila memiliki penyakit jantung, dan 0: apabila sebaliknya

Untuk memahami data lebih lanjut telah dilakukan bebera analisis seperti univariate analysis, multivariate analysis, dan beberapa visualisasi pada notebook colab.

## Data Preparation
**Menangani fitur kategorikal**

Pada dataset ini terkandung beberapa data yang bersifat kategorikal dan masih berbentuk objek sehingga sang ilmuwan perlu menangani hal ini dengan cara mengubahnya menjadi data numerik sehingga dapat dipahami oleh komputer.  Selain itu teknik yang digunakan untuk mengubah data kategorikal tersebut yaitu teknik label encoding, karena sang ilmuwan ingin melakukan eksperimen pembuatan model menggunakan algoritma tree dan setelah membaca beberapa literasi terkait teknik label encoding lebih bagus pada kasus seperti ini salah satunya karena membutuhkan lebih sedikit disk spaces.

**Membagi data latih dengan data uji**

Melakukan pembagian antara data latih dan data uji untuk keperluan melatih model nantinya.  Pembagian data latih dan uji ini menggunakan rasio 80:20 karena keseluruhan data tidak begitu banyak jumlahnya (<1000) dan rasio ini dirasa cukup ideal.

**Standarisasi data latih**

Melakukan standarisasi pada fitur numerik untuk membuat fitur data menjadi betuh yang lebih mudah diolah oleh algoritma.  Untuk menghindari kebocoran informasi pada data uji, sang ilmuwan hanya akan menerapkan fitur standarisasi pada data latih dan melakukan standarisasi pada data uji saat tahap evaluasi.

## Modelling
Pada proyek ini sang ilmuwan memutuskan untuk menggunakan 3 algoritma berbasis tree yaitu Decision Tree, Random Forest, dan XG Boost yang akan dijelaskan lebih lanjut pada pembahasan dibawah ini.

**Decision Tree**

Merupakan sebuah algoritma predictive modelling berbasis tree yang biasa digunakan pada kasus data mining dan machine learning.  Pada kasus klasifikasi membagi-membagi datasetnya berdasarkan fitur yang ingin diklasifikasikan mulai dari fitur terpenting akan menjadi root dan terbelah lagi kebawah menyesuaikan fitur-fitur penting lainnya dan pembelahan ini juga tergantung pada parameter model seperti kedalaman, dan sample pembelahan minimal.

- Kelebihan: 

1. Membutuhkan usaha yang sedikit dalam tahap data preparation timbang algoritma lainnya
2. Tidak membutuhkan normalisasi dan scaling untuk dapat berfungsi baik
3. Model yang sangat intuitif dan mudah dijelaskan kepada tim teknikal dan stakeholder


- Kekurangan:

1. Perubahan yang kecil pada data dapat menimbulkan perubahan struktur yang cukup signifikan pada model pohon
2. Perhitungan dapat lebih komplex dibandingkan algoritma lainnya
3. Relatif mahal berdasarkan kompleksitas waktu

**Random Forest**

Merupakan sebuah algoritma

- Kelebihan:


- Kekurangan:


**XGBoost**

Merupakan sebuah algoritma

- Kelebihan:

- Kekurangan: 

## Evaluation

