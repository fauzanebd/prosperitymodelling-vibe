import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the app and models
from app import create_app, db
from app.models.indicators import INDICATOR_MODELS
from app.lib import load_data, INDICATOR_PROCESSORS, label_iqr

def import_data():
    """Import data from CSV files into the database"""
    print("Importing data...")
    
    # Create app context
    app = create_app()
    with app.app_context():
        # Load data from CSV files
        data_ekonomi_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ekonomi"))
        data_infra_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/infrastruktur"))
        data_kesehatan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/kesehatan"))
        data_pendidikan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/pendidikan"))
        
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
            file_info["data"] = load_data(file_path)
        
        for indicator, file_info in data_infrastruktur_indicator_to_file.items():
            file_path = os.path.join(data_infra_dir, file_info["file"])
            file_info["data"] = load_data(file_path)
        
        for indicator, file_info in data_kesehatan_indicator_to_file.items():
            file_path = os.path.join(data_kesehatan_dir, file_info["file"])
            file_info["data"] = load_data(file_path)
        
        for indicator, file_info in data_pendidikan_indicator_to_file.items():
            file_path = os.path.join(data_pendidikan_dir, file_info["file"])
            file_info["data"] = load_data(file_path)
        
        # Process data
        all_data = {}
        
        # Process economic indicators
        for indicator, file_info in data_ekonomi_indicator_to_file.items():
            if indicator in INDICATOR_PROCESSORS:
                all_data[indicator] = INDICATOR_PROCESSORS[indicator](file_info["data"])
        
        # Process infrastructure indicators
        for indicator, file_info in data_infrastruktur_indicator_to_file.items():
            if indicator in INDICATOR_PROCESSORS:
                if indicator == 'kendaraan':
                    df_roda_2, df_roda_4 = INDICATOR_PROCESSORS[indicator](file_info["data"])
                    all_data['kendaraan_roda_2'] = df_roda_2
                    all_data['kendaraan_roda_4'] = df_roda_4
                else:
                    all_data[indicator] = INDICATOR_PROCESSORS[indicator](file_info["data"])
        
        # Process health indicators
        for indicator, file_info in data_kesehatan_indicator_to_file.items():
            if indicator in INDICATOR_PROCESSORS:
                all_data[indicator] = INDICATOR_PROCESSORS[indicator](file_info["data"])
        
        # Process education indicators
        for indicator, file_info in data_pendidikan_indicator_to_file.items():
            if indicator in INDICATOR_PROCESSORS:
                if indicator == 'angka_partisipasi_murni':
                    result_dfs = INDICATOR_PROCESSORS[indicator](file_info["data"])
                    # Add each education level's DataFrame to all_data
                    for level_name, df in result_dfs.items():
                        all_data[level_name] = df
                elif indicator == 'angka_partisipasi_kasar':
                    result_dfs = INDICATOR_PROCESSORS[indicator](file_info["data"])
                    # Add each education level's DataFrame to all_data
                    for level_name, df in result_dfs.items():
                        all_data[level_name] = df
                else:
                    all_data[indicator] = INDICATOR_PROCESSORS[indicator](file_info["data"])
        
        # Apply IQR labeling
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
            df_labeled = df.copy()
            df_labeled['label_sejahtera'] = label_iqr(df_labeled, indicator, reverse=reverse)
            all_data_final[indicator] = df_labeled
        
        # Import data into the database
        for indicator, df in all_data_final.items():
            if indicator in INDICATOR_MODELS:
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