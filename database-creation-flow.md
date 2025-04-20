# Alur Inisiasi Skema Database dalam Aplikasi Flask

## 1. Inisiasi Database di `app/__init__.py`

Alur inisiasi skema database dimulai dari file `app/__init__.py` yang merupakan titik masuk utama aplikasi. Berikut adalah tahapannya:

1. **Inisiasi Objek Database**:
   ```python
   db = SQLAlchemy()
   migrate = Migrate()
   ```
   Objek `db` adalah instance dari SQLAlchemy yang akan digunakan untuk mendefinisikan model dan berinteraksi dengan database. Objek `migrate` adalah instance dari Flask-Migrate untuk mengelola migrasi database.

2. **Konfigurasi Database dalam `create_app()`**:
   ```python
   app.config.from_mapping(
       SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/prosperity'),
       SQLALCHEMY_TRACK_MODIFICATIONS=False,
   )
   ```
   Konfigurasi ini menentukan URL koneksi database (PostgreSQL) dan pengaturan lainnya.

3. **Inisiasi SQLAlchemy dengan Aplikasi**:
   ```python
   db.init_app(app)
   migrate.init_app(app, db)
   ```
   Metode `db.init_app(app)` menghubungkan objek SQLAlchemy dengan aplikasi Flask.
   Metode `migrate.init_app(app, db)` menghubungkan Flask-Migrate dengan aplikasi dan database.

4. **Pembuatan Tabel Database**:
   ```python
   with app.app_context():
       db.create_all()
   ```
   Perintah `db.create_all()` akan membuat semua tabel berdasarkan model yang telah didefinisikan.

## 2. Model-Model di `app/models/`

Direktori `app/models/` berisi definisi model-model database yang digunakan aplikasi:

1. **`app/models/__init__.py`**:
   File ini mengimpor semua model dari modul terpisah untuk memudahkan akses:
   ```python
   from app.models.user import User
   from app.models.indicators import *
   from app.models.ml_models import TrainedModel
   from app.models.predictions import RegionPrediction
   ```

2. **`app/models/indicators.py`**:
   File ini mendefinisikan model-model indikator kesejahteraan:
   - Menggunakan model abstrak `BaseIndicator` sebagai kelas dasar
   - Mendefinisikan berbagai indikator ekonomi, infrastruktur, kesehatan, dan pendidikan
   - Setiap model indikator memetakan ke tabel database tertentu melalui atribut `__tablename__`
   - Contoh model:
     ```python
     class IndeksPembangunanManusia(BaseIndicator):
         __tablename__ = 'indeks_pembangunan_manusia'
         unit = 'persen'
         value = db.Column(db.Float)
         label_sejahtera = db.Column(db.String(64))
     ```

3. **Model Lainnya**:
   - `user.py`: Mendefinisikan model User untuk autentikasi
   - `ml_models.py`: Mendefinisikan model untuk menyimpan model machine learning
   - `predictions.py`: Mendefinisikan model untuk menyimpan hasil prediksi
   - `thresholds.py`: Mendefinisikan model untuk menyimpan nilai ambang batas

## 3. Inisialisasi Database dengan `app/migrations/init_db.py`

File `init_db.py` berperan dalam inisialisasi awal database:

1. **Impor Dependensi**:
   ```python
   from app import create_app, db
   from app.models.user import User
   from app.models.indicators import *
   from app.models.ml_models import TrainedModel
   from app.models.predictions import RegionPrediction
   ```

2. **Fungsi `init_db()`**:
   - Membuat konteks aplikasi dengan `app = create_app()`
   - Menggunakan `db.create_all()` untuk membuat semua tabel berdasarkan model
   - Membuat pengguna awal jika belum ada

## 4. Hubungan antara Flask-SQLAlchemy, Flask-Migrate, dan Model

1. **Flask-SQLAlchemy**:
   - Berfungsi sebagai ORM (Object Relational Mapper) yang memetakan kelas Python ke tabel database
   - Menyediakan objek `db` yang digunakan untuk mendefinisikan model dan berinteraksi dengan database
   - Menggunakan `db.Model` sebagai kelas dasar untuk semua model
   - Menggunakan `db.Column` untuk mendefinisikan kolom tabel

2. **Flask-Migrate**:
   - Berperan sebagai alat migrasi database yang terintegrasi dengan Flask dan SQLAlchemy
   - Memungkinkan pembuatan, penerapan, dan pengelolaan skema database secara terstruktur
   - Diinisialisasi dengan `migrate.init_app(app, db)` untuk menghubungkan dengan aplikasi dan database
   - Meskipun tidak terlihat penggunaannya secara eksplisit dalam kode yang diberikan, Flask-Migrate biasanya digunakan melalui perintah CLI seperti `flask db migrate` dan `flask db upgrade`

3. **Alur Kerja**:
   - Model-model didefinisikan menggunakan `db.Model` dari SQLAlchemy
   - Saat aplikasi dimulai, `db.init_app(app)` menghubungkan SQLAlchemy dengan aplikasi
   - `migrate.init_app(app, db)` menghubungkan Flask-Migrate dengan aplikasi dan database
   - `db.create_all()` membuat tabel berdasarkan model yang telah didefinisikan
   - Flask-Migrate dapat digunakan untuk mengelola perubahan skema database di masa mendatang

## 5. Bagaimana Kelas yang Menginherit db.Model Digunakan untuk Referensi Skema

### Peran SQLAlchemy dalam Definisi Skema

1. **Deklaratif Base Class**:
   - Ketika kelas Python menginherit dari `db.Model`, kelas tersebut menjadi bagian dari sistem "declarative base" SQLAlchemy
   - `db.Model` adalah kelas dasar yang disediakan oleh Flask-SQLAlchemy yang sudah dikonfigurasi dengan metadata SQLAlchemy

2. **Metadata dan Refleksi**:
   - Setiap kelas model memiliki metadata yang berisi informasi tentang tabel database yang terkait
   - SQLAlchemy menggunakan metadata ini untuk:
     - Menghasilkan skema SQL
     - Memetakan objek Python ke record database
     - Melakukan operasi ORM seperti query, insert, update, dan delete

3. **Atribut Kelas Menjadi Skema Database**:
   - Atribut kelas yang didefinisikan dengan `db.Column()` menjadi kolom dalam tabel database
   - Atribut khusus seperti `__tablename__` menentukan nama tabel
   - Atribut lain seperti `__abstract__ = True` menandakan bahwa kelas tersebut adalah kelas abstrak yang tidak memiliki tabel sendiri

4. **Relasi antar Model**:
   - SQLAlchemy mendukung definisi relasi antar model menggunakan `db.relationship()`
   - Relasi ini kemudian diterjemahkan menjadi foreign key dan constraint dalam skema database

### Peran Flask-Migrate dalam Pengelolaan Skema

1. **Integrasi dengan Alembic**:
   - Flask-Migrate adalah wrapper untuk Alembic, library migrasi database untuk SQLAlchemy
   - Alembic menyediakan infrastruktur untuk mengelola perubahan skema database secara terstruktur

2. **Deteksi Perubahan Model**:
   - Flask-Migrate membandingkan definisi model dalam kode Python dengan skema database yang ada
   - Perbedaan yang terdeteksi digunakan untuk menghasilkan skrip migrasi

3. **Proses Migrasi**:
   - Saat menjalankan `flask db migrate`, Flask-Migrate:
     1. Memuat semua model yang menginherit dari `db.Model`
     2. Membandingkan definisi model dengan skema database saat ini
     3. Menghasilkan skrip migrasi yang berisi perubahan yang diperlukan

4. **Penerapan Migrasi**:
   - Saat menjalankan `flask db upgrade`, Flask-Migrate:
     1. Menjalankan skrip migrasi yang belum diterapkan
     2. Memperbarui skema database sesuai dengan definisi model terbaru
     3. Mencatat migrasi yang telah diterapkan dalam tabel `alembic_version`

### Perbedaan antara db.create_all() dan Flask-Migrate

1. **db.create_all()**:
   - Metode sederhana untuk membuat tabel berdasarkan model
   - Hanya membuat tabel yang belum ada
   - Tidak melacak perubahan skema atau memperbarui tabel yang sudah ada
   - Cocok untuk pengembangan awal atau aplikasi sederhana

2. **Flask-Migrate**:
   - Solusi yang lebih canggih untuk mengelola skema database
   - Melacak perubahan skema dan memperbarui tabel yang sudah ada
   - Mendukung rollback ke versi skema sebelumnya
   - Cocok untuk aplikasi produksi yang berkembang dari waktu ke waktu

Dalam aplikasi ini, keduanya digunakan:
- `db.create_all()` dalam `create_app()` dan `init_db.py` untuk inisialisasi awal
- Flask-Migrate (melalui `migrate.init_app(app, db)`) untuk mengelola perubahan skema di masa mendatang

Pendekatan ini memungkinkan pengembangan cepat pada tahap awal sambil mempertahankan kemampuan untuk mengelola perubahan skema secara terstruktur saat aplikasi berkembang.
