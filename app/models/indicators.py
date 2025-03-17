from app import db

# Base model for all indicators
class BaseIndicator(db.Model):
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(64), index=True)
    year = db.Column(db.Integer, index=True)
    
    def __repr__(self):
        return f'<{self.__class__.__name__} {self.region} {self.year}>'

# Economic Indicators
class IndeksPembangunanManusia(BaseIndicator):
    __tablename__ = 'indeks_pembangunan_manusia'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class TingkatPengangguranTerbuka(BaseIndicator):
    __tablename__ = 'tingkat_pengangguran_terbuka'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class PdrbHargaKonstan(BaseIndicator):
    __tablename__ = 'pdrb_harga_konstan'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class PendudukMiskin(BaseIndicator):
    __tablename__ = 'penduduk_miskin'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class JmlPengeluaranPerKapita(BaseIndicator):
    __tablename__ = 'jml_pengeluaran_per_kapita'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class JmlPendudukBekerja(BaseIndicator):
    __tablename__ = 'jml_penduduk_bekerja'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class DaftarUpahMinimum(BaseIndicator):
    __tablename__ = 'daftar_upah_minimum'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Infrastructure Indicators
class SanitasiLayak(BaseIndicator):
    __tablename__ = 'sanitasi_layak'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class HunianLayak(BaseIndicator):
    __tablename__ = 'hunian_layak'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AksesAirMinum(BaseIndicator):
    __tablename__ = 'akses_air_minum'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class KawasanPariwisata(BaseIndicator):
    __tablename__ = 'kawasan_pariwisata'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Replace Kendaraan class with Roda2 and Roda4
class KendaraanRoda2(BaseIndicator):
    __tablename__ = 'kendaraan_roda_2'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class KendaraanRoda4(BaseIndicator):
    __tablename__ = 'kendaraan_roda_4'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class PanjangRuasJalan(BaseIndicator):
    __tablename__ = 'panjang_ruas_jalan'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class TitikLayananInternet(BaseIndicator):
    __tablename__ = 'titik_layanan_internet'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Health Indicators
class AngkaHarapanHidup(BaseIndicator):
    __tablename__ = 'angka_harapan_hidup'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class FasilitasKesehatan(BaseIndicator):
    __tablename__ = 'fasilitas_kesehatan'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class KematianBalita(BaseIndicator):
    __tablename__ = 'kematian_balita'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class KematianBayi(BaseIndicator):
    __tablename__ = 'kematian_bayi'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class KematianIbu(BaseIndicator):
    __tablename__ = 'kematian_ibu'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class PersentaseBalitaStunting(BaseIndicator):
    __tablename__ = 'persentase_balita_stunting'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class ImunisasiDasar(BaseIndicator):
    __tablename__ = 'imunisasi_dasar'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Education Indicators
class AngkaMelekHuruf(BaseIndicator):
    __tablename__ = 'angka_melek_huruf'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Replace AngkaPartisipasiKasar with specific education levels
class AngkaPartisipasiKasarSD(BaseIndicator):
    __tablename__ = 'angka_partisipasi_kasar_sd_mi_paket_a'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AngkaPartisipasiKasarSMP(BaseIndicator):
    __tablename__ = 'angka_partisipasi_kasar_smp_mts_paket_b'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AngkaPartisipasiKasarSMA(BaseIndicator):
    __tablename__ = 'angka_partisipasi_kasar_sma_ma_paket_c'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AngkaPartisipasiKasarPT(BaseIndicator):
    __tablename__ = 'angka_partisipasi_kasar_perguruan_tinggi'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Replace AngkaPartisipasiMurni with specific education levels
class AngkaPartisipasiMurniSD(BaseIndicator):
    __tablename__ = 'angka_partisipasi_murni_sd_mi_paket_a'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AngkaPartisipasiMurniSMP(BaseIndicator):
    __tablename__ = 'angka_partisipasi_murni_smp_mts_paket_b'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AngkaPartisipasiMurniSMA(BaseIndicator):
    __tablename__ = 'angka_partisipasi_murni_sma_ma_paket_c'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class AngkaPartisipasiMurniPT(BaseIndicator):
    __tablename__ = 'angka_partisipasi_murni_perguruan_tinggi'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

class RataRataLamaSekolah(BaseIndicator):
    __tablename__ = 'rata_rata_lama_sekolah'
    
    value = db.Column(db.Float)
    label_sejahtera = db.Column(db.String(64))

# Update the INDICATOR_MODELS dictionary
INDICATOR_MODELS = {
    # Economic indicators
    'indeks_pembangunan_manusia': IndeksPembangunanManusia,
    'tingkat_pengangguran_terbuka': TingkatPengangguranTerbuka,
    'pdrb_harga_konstan': PdrbHargaKonstan,
    'penduduk_miskin': PendudukMiskin,
    'jml_pengeluaran_per_kapita': JmlPengeluaranPerKapita,
    'jml_penduduk_bekerja': JmlPendudukBekerja,
    'daftar_upah_minimum': DaftarUpahMinimum,
    
    # Infrastructure indicators
    'sanitasi_layak': SanitasiLayak,
    'hunian_layak': HunianLayak,
    'akses_air_minum': AksesAirMinum,
    'kawasan_pariwisata': KawasanPariwisata,
    'kendaraan_roda_2': KendaraanRoda2,
    'kendaraan_roda_4': KendaraanRoda4,
    'panjang_ruas_jalan': PanjangRuasJalan,
    'titik_layanan_internet': TitikLayananInternet,
    
    # Health indicators
    'angka_harapan_hidup': AngkaHarapanHidup,
    'fasilitas_kesehatan': FasilitasKesehatan,
    'kematian_balita': KematianBalita,
    'kematian_bayi': KematianBayi,
    'kematian_ibu': KematianIbu,
    'persentase_balita_stunting': PersentaseBalitaStunting,
    'imunisasi_dasar': ImunisasiDasar,
    
    # Education indicators
    'angka_melek_huruf': AngkaMelekHuruf,
    'angka_partisipasi_kasar_sd_mi_paket_a': AngkaPartisipasiKasarSD,
    'angka_partisipasi_kasar_smp_mts_paket_b': AngkaPartisipasiKasarSMP,
    'angka_partisipasi_kasar_sma_ma_paket_c': AngkaPartisipasiKasarSMA,
    'angka_partisipasi_kasar_perguruan_tinggi': AngkaPartisipasiKasarPT,
    'angka_partisipasi_murni_sd_mi_paket_a': AngkaPartisipasiMurniSD,
    'angka_partisipasi_murni_smp_mts_paket_b': AngkaPartisipasiMurniSMP,
    'angka_partisipasi_murni_sma_ma_paket_c': AngkaPartisipasiMurniSMA,
    'angka_partisipasi_murni_perguruan_tinggi': AngkaPartisipasiMurniPT,
    'rata_rata_lama_sekolah': RataRataLamaSekolah
} 