# Hiring Decision Analysis

## Tujuan Analysis 

  Proyek ini bertujuan untuk menganalisis dan memprediksi keputusan perekrutan menggunakan model regresi logistik. Dalam studi ini, saya mengeksplorasi berbagai faktor yang berpotensi memengaruhi hasil keputusan perekrutan, seperti usia, jenis kelamin, tingkat pendidikan, jumlah tahun pengalaman kerja, riwayat perusahaan sebelumnya, jarak tempat tinggal ke perusahaan, skor wawancara, skor keterampilan teknis, skor kepribadian, serta strategi perekrutan yang digunakan.

## Deskripsi Data

Dataset terdiri dari kolom-kolom berikut:

### Deskripsi Variabel

  - **Jenis Kelamin**:
    - Categorical : Laki-laki atau perempuan
    - Data Type   : Binary
   
  - **python_exp**:
     - Categorical : yes atau no
     - Data Type   : Binary 
       
  - **Experience Years**:
    -  Data Range  : 0 sampai 3 tahun
    -  Data Type   : Integer
    
  - **Education Level**:
    - Categorical 1 : Graduate tipe (1) atau Not Graduate tipe (1)
    - Data Tyoe     : Categori
  
  - **Internship**:
    - Categorical : yes or no
    - Data Type   : Binary
      
 - **Skil Score** :
   - Data Range   : 0 sampai 20000
   - Data Type    : Integer
  
- **Sallary 10*E4**:
  - Data Range    : 0 sampai 500
  - Data Type     : Integer

- **Offer_History**:
  - Data Range    : 0 sampai 1
  - Data Type     : Integer
 
- **location**     :
  - Categorical   : Alamat
  - Data Type     : String

- **Recuitmen_Status**  :
  - Categorical   : Y atau N
  - Data Type     : String

## Exploratory Data Analysis (EDA)

### Analisis Korelasi
Langkah awal dalam analisi ini adalah melakukan evaluasi korelasi antar fitur terhadap variabel target, Keputasan Perekrutan. Dari sini, kami dapat mengetahui fitur mana saja yang memiliki konntribusi yang signifikan terhadap hasil akhir perekrutan. 

## Predictive Modeling
### Logistic Regression
Dalam proyek ini, kami memilih logistic Regression sebagai algoritma utama karena cocok digunakan untuk klasifikasi biner. variabel target, yaitu keputusan perekrutan, memiliki dua label output : label ini dipekerjakan atau tidak di pekerjakan. model ini membantu mematakan hubungan antara fitur-fitu input dan kemungkinan seseorang yang diterima kerja. 



