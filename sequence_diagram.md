# Kesejahteraan Application Sequence Diagram

## User Authentication Flow

```plantuml
@startuml Authentication
actor Pengguna
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Manajer Masuk" as LoginManager
participant "Model Pengguna" as UserModel
database "Basis Data" as DB

Pengguna -> Browser: Akses aplikasi
Browser -> Flask: GET /
Flask -> LoginManager: Periksa autentikasi
LoginManager -> UserModel: apakah terautentikasi?

alt Belum terautentikasi
    UserModel --> LoginManager: Tidak
    LoginManager -> UserModel: Cari pengguna default
    UserModel -> DB: SELECT FROM users WHERE username='pengunjung'
    DB --> UserModel: Kembalikan data pengguna
    UserModel -> LoginManager: Buat sesi pengguna
    LoginManager --> Flask: Pengguna terautentikasi sebagai 'pengunjung'
else Sudah terautentikasi
    UserModel --> LoginManager: Ya
    LoginManager --> Flask: Pengguna sudah terautentikasi
end

Flask -> Browser: Render dashboard
Browser --> Pengguna: Tampilkan dashboard
@enduml
```

## Admin Login Flow

```plantuml
@startuml AdminLogin
actor Admin
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Pengendali Autentikasi" as AuthController
participant "Model Pengguna" as UserModel
database "Basis Data" as DB

Admin -> Browser: Klik 'Beralih ke Admin'
Browser -> Flask: GET /switch-to-admin
Flask -> AuthController: Tangani permintaan
AuthController -> Browser: Render formulir login admin

Admin -> Browser: Masukkan kredensial admin
Browser -> Flask: POST /switch-to-admin
Flask -> AuthController: Tangani permintaan login
AuthController -> UserModel: Cari pengguna admin
UserModel -> DB: SELECT FROM users WHERE username=? AND is_admin=true
DB --> UserModel: Kembalikan data pengguna admin

AuthController -> UserModel: check_password()
alt Kredensial valid
    UserModel --> AuthController: Kata sandi benar
    AuthController -> LoginManager: Logout pengguna saat ini
    AuthController -> LoginManager: Login pengguna admin
    AuthController -> Browser: Alihkan ke dashboard
    Browser --> Admin: Tampilkan dashboard sebagai admin
else Kredensial tidak valid
    UserModel --> AuthController: Kata sandi salah
    AuthController -> Browser: Tampilkan pesan kesalahan
    Browser --> Admin: Tampilkan formulir login dengan kesalahan
end
@enduml
```

## Dataset Visualization Flow

```plantuml
@startuml DataVisualization
actor Pengguna
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Pengendali Visualisasi" as VisController
participant "Layanan Data" as DataServices
participant "Layanan Visualisasi" as VisServices
database "Basis Data" as DB

Pengguna -> Browser: Akses halaman visualisasi
Browser -> Flask: GET /visualization/data
Flask -> VisController: Tangani permintaan
VisController -> DB: Cari data indikator
DB --> VisController: Kembalikan data indikator

VisController -> DataServices: Proses data
DataServices --> VisController: Data terproses
VisController -> VisServices: Hasilkan plot visualisasi
VisServices --> VisController: Kembalikan JSON visualisasi

VisController -> Browser: Render halaman visualisasi dengan data
Browser --> Pengguna: Tampilkan visualisasi

Pengguna -> Browser: Interaksi dengan visualisasi (filter, ubah parameter)
Browser -> Flask: Permintaan AJAX dengan parameter baru
Flask -> VisController: Proses pembaruan visualisasi
VisController -> DB: Cari dengan parameter baru
DB --> VisController: Kembalikan data terfilter
VisController -> VisServices: Hasilkan visualisasi terbaru
VisServices --> VisController: Kembalikan JSON terbaru
VisController -> Browser: Kembalikan respons JSON
Browser -> Browser: Perbarui visualisasi secara dinamis
Browser --> Pengguna: Tampilkan visualisasi terbaru
@enduml
```

## Model Training Flow

```plantuml
@startuml ModelTraining
actor Admin
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Pengendali Dataset" as DatasetController
participant "Layanan Pelatihan Model" as ModelTrainer
participant "Layanan Pemrosesan Data" as DataProcessor
database "Basis Data" as DB

Admin -> Browser: Akses halaman pelatihan model
Browser -> Flask: GET /dataset/train-models
Flask -> DatasetController: Tangani permintaan
DatasetController -> Browser: Render formulir pelatihan model

Admin -> Browser: Konfigurasikan dan kirim pelatihan
Browser -> Flask: POST /dataset/train-models
Flask -> DatasetController: Proses permintaan pelatihan

DatasetController -> DB: Cari dataset
DB --> DatasetController: Kembalikan dataset

DatasetController -> DataProcessor: Pra-proses data
DataProcessor --> DatasetController: Data telah dipra-proses

DatasetController -> ModelTrainer: Latih model
ModelTrainer -> ModelTrainer: Latih dan validasi model
ModelTrainer -> DB: Simpan model terlatih
DB --> ModelTrainer: Konfirmasi penyimpanan

ModelTrainer -> ModelTrainer: Hasilkan prediksi
ModelTrainer -> DB: Simpan prediksi
DB --> ModelTrainer: Konfirmasi penyimpanan

ModelTrainer --> DatasetController: Pelatihan selesai
DatasetController -> Browser: Alihkan ke halaman performa model
Browser --> Admin: Tampilkan performa model
@enduml
```

## Dashboard Flow

```plantuml
@startuml Dashboard
actor Pengguna
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Pengendali Dashboard" as DashboardController
participant "Model Prediksi" as PredictionModel
database "Basis Data" as DB

Pengguna -> Browser: Akses dashboard
Browser -> Flask: GET /
Flask -> DashboardController: Tangani permintaan
DashboardController -> DB: Cari model terbaik
DB --> DashboardController: Kembalikan model terbaik

DashboardController -> DB: Cari prediksi untuk tahun terpilih
DB --> DashboardController: Kembalikan data prediksi

DashboardController -> DashboardController: Proses statistik prediksi
DashboardController -> Browser: Render dashboard dengan data dan peta
Browser --> Pengguna: Tampilkan dashboard dengan peta kesejahteraan

Pengguna -> Browser: Ubah filter tahun
Browser -> Flask: GET /?year=<tahun_terpilih>
Flask -> DashboardController: Tangani permintaan dengan parameter tahun
DashboardController -> DB: Cari prediksi untuk tahun baru
DB --> DashboardController: Kembalikan data prediksi terfilter
DashboardController -> Browser: Render dashboard yang diperbarui
Browser --> Pengguna: Tampilkan dashboard yang diperbarui
@enduml
```

## Data Addition Flow

```plantuml
@startuml DataAddition
actor Admin
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Pengendali Dataset" as DatasetController
participant "Pemroses Data" as DataProcessor
participant "Pelatih Model" as ModelTrainer
database "Basis Data" as DB

Admin -> Browser: Akses formulir penambahan data
Browser -> Flask: GET /dataset/add-for-training
Flask -> DatasetController: Tangani permintaan
DatasetController -> Browser: Render formulir data

Admin -> Browser: Isi formulir dan kirim
Browser -> Flask: POST /dataset/add-for-training
Flask -> DatasetController: Proses pengiriman

DatasetController -> DataProcessor: Pra-proses data yang dikirim
DataProcessor --> DatasetController: Data terproses

loop Untuk setiap indikator
    DatasetController -> DB: Simpan data indikator
    DB --> DatasetController: Konfirmasi penyimpanan
end

DatasetController -> ModelTrainer: Periksa apakah pelatihan ulang diperlukan
alt Pelatihan ulang diperlukan
    ModelTrainer -> DB: Cari dataset yang diperbarui
    DB --> ModelTrainer: Kembalikan dataset
    ModelTrainer -> ModelTrainer: Latih ulang model
    ModelTrainer -> DB: Simpan model baru
    DB --> ModelTrainer: Konfirmasi penyimpanan
    ModelTrainer -> ModelTrainer: Hasilkan prediksi
    ModelTrainer -> DB: Simpan prediksi
    DB --> ModelTrainer: Konfirmasi penyimpanan
end

DatasetController -> Browser: Alihkan ke halaman dataset
Browser --> Admin: Tampilkan halaman dataset dengan pesan sukses
@enduml
```

## Alur Penambahan Data untuk Inferensi

```plantuml
@startuml DataInference
actor Admin
participant "Peramban Web" as Browser
participant "Aplikasi Flask" as Flask
participant "Pengendali Dataset" as DatasetController
participant "Pemroses Data" as DataProcessor
participant "Model Terlatih" as TrainedModel
participant "Layanan Prediksi" as PredictionService
database "Basis Data" as DB

Admin -> Browser: Akses formulir penambahan data untuk inferensi
Browser -> Flask: GET /dataset/add-for-inference
Flask -> DatasetController: Tangani permintaan
DatasetController -> Browser: Render formulir data inferensi

Admin -> Browser: Isi wilayah, tahun dan nilai indikator
Browser -> Flask: POST /dataset/add-for-inference
Flask -> DatasetController: Proses pengiriman data

DatasetController -> DataProcessor: Pra-proses data yang dikirim
DataProcessor --> DatasetController: Data terproses dengan label inferensi

loop Untuk setiap indikator
    DatasetController -> DB: Simpan data indikator dengan label
    DB --> DatasetController: Konfirmasi penyimpanan
end

DatasetController -> TrainedModel: Dapatkan model terbaik
TrainedModel -> DB: SELECT * FROM trained_models ORDER BY accuracy DESC LIMIT 1
DB --> TrainedModel: Kembalikan model terbaik

alt Model terbaik ditemukan
    DatasetController -> PredictionService: Buat dataset gabungan untuk wilayah
    PredictionService -> DB: Kumpulkan semua data indikator untuk wilayah
    DB --> PredictionService: Kembalikan data indikator wilayah

    PredictionService -> TrainedModel: Muat model, scaler, dan nama fitur
    TrainedModel --> PredictionService: Model, scaler, dan nama fitur dimuat

    PredictionService -> DataProcessor: Siapkan data untuk prediksi
    DataProcessor --> PredictionService: Data siap untuk prediksi

    PredictionService -> PredictionService: Standardisasi fitur dengan scaler
    PredictionService -> PredictionService: Buat prediksi dengan model terlatih

    PredictionService -> DB: Simpan hasil prediksi untuk wilayah
    DB --> PredictionService: Konfirmasi penyimpanan prediksi

    PredictionService --> DatasetController: Prediksi selesai
else Model tidak ditemukan
    TrainedModel --> DatasetController: Model tidak ditemukan
end

DatasetController -> Browser: Alihkan ke halaman hasil prediksi
Browser --> Admin: Tampilkan halaman hasil prediksi dengan pesan sukses
@enduml
```
