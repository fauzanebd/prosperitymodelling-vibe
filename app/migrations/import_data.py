import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import importlib.util

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the prosperityModelling module
spec = importlib.util.spec_from_file_location("prosperityModelling", "../../prosperityModelling.py")
prosperity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prosperity_module)

# Import the app and models
from app import create_app, db
from app.models.indicators import INDICATOR_MODELS

def import_data():
    """Import data from the original notebook into the database"""
    print("Importing data from the original notebook...")
    
    # Create app context
    app = create_app()
    with app.app_context():
        # Load data from the original notebook
        data_ekonomi_dir = "../../data/ekonomi"
        data_infra_dir = "../../data/infrastruktur"
        data_kesehatan_dir = "../../data/kesehatan"
        data_pendidikan_dir = "../../data/pendidikan"
        
        # Define dictionaries to store indicator data
        data_ekonomi_indicator_to_file = {
            "indeks_pembangunan_manusia": {"file": "indeks_pembangunan_manusia.csv", "data": pd.DataFrame()},
            "tingkat_pengangguran_terbuka": {"file": "tingkat_pengangguran_terbuka.csv", "data": pd.DataFrame()},
            "pdrb_harga_konstan": {"file": "pdrb_harga_konstan.csv", "data": pd.DataFrame()},
            "penduduk_miskin": {"file": "penduduk_miskin.csv", "data": pd.DataFrame()},
            "jml_pengeluaran_per_kapita": {"file": "jml_pengeluaran_per_kapita.csv", "data": pd.DataFrame()},
            "jml_penduduk_bekerja": {"file": "jml_penduduk_bekerja.csv", "data": pd.DataFrame()},
            "daftar_upah_minimum": {"file": "daftar_upah_minimum.csv", "data": pd.DataFrame()}
        }
        
        data_infrastruktur_indicator_to_file = {
            "sanitasi_layak": {"file": "sanitasi_layak.csv", "data": pd.DataFrame()},
            "hunian_layak": {"file": "hunian_layak.csv", "data": pd.DataFrame()},
            "akses_air_minum": {"file": "akses_air_minum.csv", "data": pd.DataFrame()},
            "kawasan_pariwisata": {"file": "kawasan_pariwisata.csv", "data": pd.DataFrame()},
            "kendaraan": {"file": "kendaraan.csv", "data": pd.DataFrame()},
            "panjang_ruas_jalan": {"file": "panjang_ruas_jalan.csv", "data": pd.DataFrame()},
            "titik_layanan_internet": {"file": "titik_layanan_internet.csv", "data": pd.DataFrame()}
        }
        
        data_kesehatan_indicator_to_file = {
            "angka_harapan_hidup": {"file": "angka_harapan_hidup.csv", "data": pd.DataFrame()},
            "fasilitas_kesehatan": {"file": "fasilitas_kesehatan.csv", "data": pd.DataFrame()},
            "kematian_balita": {"file": "kematian_balita.csv", "data": pd.DataFrame()},
            "kematian_bayi": {"file": "kematian_bayi.csv", "data": pd.DataFrame()},
            "kematian_ibu": {"file": "kematian_ibu.csv", "data": pd.DataFrame()},
            "persentase_balita_stunting": {"file": "persentase_balita_stunting.csv", "data": pd.DataFrame()},
            "imunisasi_dasar": {"file": "imunisasi_dasar.csv", "data": pd.DataFrame()}
        }
        
        data_pendidikan_indicator_to_file = {
            "angka_melek_huruf": {"file": "angka_melek_huruf.csv", "data": pd.DataFrame()},
            "angka_partisipasi_kasar": {"file": "angka_partisipasi_kasar.csv", "data": pd.DataFrame()},
            "angka_partisipasi_murni": {"file": "angka_partisipasi_murni.csv", "data": pd.DataFrame()},
            "rata_rata_lama_sekolah": {"file": "rata_rata_lama_sekolah.csv", "data": pd.DataFrame()}
        }
        
        # Load data
        for indicator, file_info in data_ekonomi_indicator_to_file.items():
            file_path = os.path.join(data_ekonomi_dir, file_info["file"])
            file_info["data"] = prosperity_module.load_data(file_path)
        
        for indicator, file_info in data_infrastruktur_indicator_to_file.items():
            file_path = os.path.join(data_infra_dir, file_info["file"])
            file_info["data"] = prosperity_module.load_data(file_path)
        
        for indicator, file_info in data_kesehatan_indicator_to_file.items():
            file_path = os.path.join(data_kesehatan_dir, file_info["file"])
            file_info["data"] = prosperity_module.load_data(file_path)
        
        for indicator, file_info in data_pendidikan_indicator_to_file.items():
            file_path = os.path.join(data_pendidikan_dir, file_info["file"])
            file_info["data"] = prosperity_module.load_data(file_path)
        
        # Preprocess data
        # Economic indicators
        data_ekonomi_indicator_to_file["indeks_pembangunan_manusia"]["data"] = prosperity_module.preprocess_indeks_pembangunan_manusia(data_ekonomi_indicator_to_file["indeks_pembangunan_manusia"]["data"])
        data_ekonomi_indicator_to_file["tingkat_pengangguran_terbuka"]["data"] = prosperity_module.preprocess_tingkat_pengangguran_terbuka(data_ekonomi_indicator_to_file["tingkat_pengangguran_terbuka"]["data"])
        data_ekonomi_indicator_to_file["pdrb_harga_konstan"]["data"] = prosperity_module.preprocess_pdrb_harga_konstan(data_ekonomi_indicator_to_file["pdrb_harga_konstan"]["data"])
        data_ekonomi_indicator_to_file["penduduk_miskin"]["data"] = prosperity_module.preprocess_penduduk_miskin(data_ekonomi_indicator_to_file["penduduk_miskin"]["data"])
        data_ekonomi_indicator_to_file["jml_pengeluaran_per_kapita"]["data"] = prosperity_module.preprocess_jml_pengeluaran_per_kapita(data_ekonomi_indicator_to_file["jml_pengeluaran_per_kapita"]["data"])
        data_ekonomi_indicator_to_file["jml_penduduk_bekerja"]["data"] = prosperity_module.preprocess_jml_penduduk_bekerja(data_ekonomi_indicator_to_file["jml_penduduk_bekerja"]["data"])
        data_ekonomi_indicator_to_file["daftar_upah_minimum"]["data"] = prosperity_module.preprocess_daftar_upah_minimum(data_ekonomi_indicator_to_file["daftar_upah_minimum"]["data"])
        
        # Infrastructure indicators
        data_infrastruktur_indicator_to_file["sanitasi_layak"]["data"] = prosperity_module.preprocess_sanitasi_layak(data_infrastruktur_indicator_to_file["sanitasi_layak"]["data"])
        data_infrastruktur_indicator_to_file["hunian_layak"]["data"] = prosperity_module.preprocess_hunian_layak(data_infrastruktur_indicator_to_file["hunian_layak"]["data"])
        data_infrastruktur_indicator_to_file["akses_air_minum"]["data"] = prosperity_module.preprocess_akses_air_minum(data_infrastruktur_indicator_to_file["akses_air_minum"]["data"])
        data_infrastruktur_indicator_to_file["kawasan_pariwisata"]["data"] = prosperity_module.preprocess_kawasan_pariwisata(data_infrastruktur_indicator_to_file["kawasan_pariwisata"]["data"])
        data_infrastruktur_indicator_to_file["kendaraan"]["data"] = prosperity_module.preprocess_kendaraan(data_infrastruktur_indicator_to_file["kendaraan"]["data"])
        data_infrastruktur_indicator_to_file["panjang_ruas_jalan"]["data"] = prosperity_module.preprocess_panjang_ruas_jalan(data_infrastruktur_indicator_to_file["panjang_ruas_jalan"]["data"])
        data_infrastruktur_indicator_to_file["titik_layanan_internet"]["data"] = prosperity_module.preprocess_titik_layanan_internet(data_infrastruktur_indicator_to_file["titik_layanan_internet"]["data"])
        
        # Health indicators
        data_kesehatan_indicator_to_file["angka_harapan_hidup"]["data"] = prosperity_module.preprocess_angka_harapan_hidup(data_kesehatan_indicator_to_file["angka_harapan_hidup"]["data"])
        data_kesehatan_indicator_to_file["fasilitas_kesehatan"]["data"] = prosperity_module.preprocess_fasilitas_kesehatan(data_kesehatan_indicator_to_file["fasilitas_kesehatan"]["data"])
        data_kesehatan_indicator_to_file["kematian_balita"]["data"] = prosperity_module.preprocess_kematian_balita(data_kesehatan_indicator_to_file["kematian_balita"]["data"])
        data_kesehatan_indicator_to_file["kematian_bayi"]["data"] = prosperity_module.preprocess_kematian_bayi(data_kesehatan_indicator_to_file["kematian_bayi"]["data"])
        data_kesehatan_indicator_to_file["kematian_ibu"]["data"] = prosperity_module.preprocess_kematian_ibu(data_kesehatan_indicator_to_file["kematian_ibu"]["data"])
        data_kesehatan_indicator_to_file["persentase_balita_stunting"]["data"] = prosperity_module.preprocess_persentase_balita_stunting(data_kesehatan_indicator_to_file["persentase_balita_stunting"]["data"])
        data_kesehatan_indicator_to_file["imunisasi_dasar"]["data"] = prosperity_module.preprocess_imunisasi_dasar(data_kesehatan_indicator_to_file["imunisasi_dasar"]["data"])
        
        # Education indicators
        data_pendidikan_indicator_to_file["angka_melek_huruf"]["data"] = prosperity_module.preprocess_angka_melek_huruf(data_pendidikan_indicator_to_file["angka_melek_huruf"]["data"])
        data_pendidikan_indicator_to_file["angka_partisipasi_kasar"]["data"] = prosperity_module.preprocess_angka_partisipasi_kasar(data_pendidikan_indicator_to_file["angka_partisipasi_kasar"]["data"])
        data_pendidikan_indicator_to_file["angka_partisipasi_murni"]["data"] = prosperity_module.preprocess_angka_partisipasi_murni(data_pendidikan_indicator_to_file["angka_partisipasi_murni"]["data"])
        data_pendidikan_indicator_to_file["rata_rata_lama_sekolah"]["data"] = prosperity_module.preprocess_rata_rata_lama_sekolah(data_pendidikan_indicator_to_file["rata_rata_lama_sekolah"]["data"])
        
        # Combine all indicators
        all_data = {}
        all_data.update({k: v["data"] for k, v in data_ekonomi_indicator_to_file.items()})
        all_data.update({k: v["data"] for k, v in data_infrastruktur_indicator_to_file.items()})
        all_data.update({k: v["data"] for k, v in data_kesehatan_indicator_to_file.items()})
        all_data.update({k: v["data"] for k, v in data_pendidikan_indicator_to_file.items()})
        
        # Apply IQR labeling to all indicators
        all_data_final = {}
        for indicator, df in all_data.items():
            # Special case for indicators where lower values are better
            reverse = indicator in [
                'tingkat_pengangguran_terbuka',
                'penduduk_miskin',
                'kematian_balita',
                'kematian_bayi',
                'kematian_ibu',
                'persentase_balita_stunting'
            ]
            
            # Apply IQR labeling
            df_labeled = prosperity_module.label_iqr(df, indicator, reverse=reverse)
            all_data_final[indicator] = df_labeled
        
        # Import data into the database
        for indicator, df in all_data_final.items():
            model_class = INDICATOR_MODELS[indicator]
            
            # Delete existing data
            model_class.query.delete()
            
            # Insert new data
            for _, row in df.iterrows():
                new_data = model_class(
                    provinsi=row['wilayah'],
                    year=row['year'],
                    value=row[indicator],
                    label_sejahtera=row['label_sejahtera']
                )
                db.session.add(new_data)
            
            db.session.commit()
            print(f"Imported {len(df)} rows for {indicator}")
        
        print("Data import completed successfully!")

if __name__ == "__main__":
    import_data() 