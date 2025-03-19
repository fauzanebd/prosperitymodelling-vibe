import pandas as pd
from .data_processor import preprocess_yearly_data, preprocess_standard_data

# Infrastructure indicators preprocessing functions
def preprocess_akses_air_minum(df):
    """Preprocess akses_air_minum data"""
    return preprocess_yearly_data(df, "akses_air_minum")

def preprocess_hunian_layak(df):
    """Preprocess hunian_layak data"""
    return preprocess_yearly_data(df, "hunian_layak")

def preprocess_kawasan_pariwisata(df):
    """Preprocess kawasan_pariwisata data"""
    return preprocess_standard_data(df, "kawasan_pariwisata")

def preprocess_kendaraan(df):
    """
    Preprocess kendaraan data, splitting into kendaraan_roda_2 and kendaraan_roda_4
    """
    # Filter data for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    type_col = [col for col in df.columns if 'tipe' in col.lower() or 'roda' in col.lower()][0]
    value_col = [col for col in df.columns if 'jumlah' in col.lower()][0]
    
    # Split data by vehicle type
    df_roda_2 = df_filtered[df_filtered[type_col].str.contains('2', na=False)].copy()
    df_roda_4 = df_filtered[df_filtered[type_col].str.contains('4', na=False)].copy()
    
    # Process each dataset
    df_roda_2 = df_roda_2.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'kendaraan_roda_2'
    })
    
    df_roda_4 = df_roda_4.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'kendaraan_roda_4'
    })
    
    # Select only required columns
    df_roda_2 = df_roda_2[['wilayah', 'kendaraan_roda_2', 'year']].copy()
    df_roda_4 = df_roda_4[['wilayah', 'kendaraan_roda_4', 'year']].copy()
    
    # Convert values to numeric
    df_roda_2['kendaraan_roda_2'] = pd.to_numeric(df_roda_2['kendaraan_roda_2'], errors='coerce')
    df_roda_4['kendaraan_roda_4'] = pd.to_numeric(df_roda_4['kendaraan_roda_4'], errors='coerce')
    
    return df_roda_2, df_roda_4

def preprocess_panjang_ruas_jalan(df):
    """
    Preprocess panjang_ruas_jalan data, summing different wilayah_uptd for the same region
    """
    # Filter for years 2019-2023
    target_years = ['2019', '2020', '2021', '2022', '2023']
    
    # Clean the year column by removing decimal part
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    
    # Split at comma and take the first part, then remove any decimal point
    df[year_col] = df[year_col].astype(str).str.split(',').str[0].str.split('.').str[0]
    
    # Check if there's any data for years 2019-2023
    available_years = df[year_col].unique()
    years_present = [year for year in target_years if year in available_years]
    
    if not years_present:
        print(f"Warning: No data found for years 2019-2023 in panjang_ruas_jalan. Available years: {available_years}")
        print("Using the most recent available year as a substitute.")
        
        # Sort available years and use the most recent one
        available_years_sorted = sorted(available_years)
        most_recent_year = available_years_sorted[-1]
        print(f"Using {most_recent_year} data for all years 2019-2023")
        
        # Create a copy of the dataframe with the most recent year
        df_most_recent = df[df[year_col] == most_recent_year].copy()
        
        # Create copies for each target year
        result_dfs = []
        for year in target_years:
            df_year = df_most_recent.copy()
            df_year[year_col] = year
            result_dfs.append(df_year)
        
        # Combine all years
        df = pd.concat(result_dfs, ignore_index=True)
    else:
        # Filter for target years
        df = df[df[year_col].isin(target_years)].copy()
        
        # If some years are missing, fill them with the most recent available year
        missing_years = [year for year in target_years if year not in years_present]
        if missing_years:
            print(f"Warning: Missing years {missing_years} in panjang_ruas_jalan data. Using most recent available year as substitute.")
            
            # Sort available years and use the most recent one
            years_present_sorted = sorted(years_present)
            most_recent_year = years_present_sorted[-1]
            print(f"Using {most_recent_year} data for missing years")
            
            # Create copies for each missing year
            result_dfs = [df]
            for year in missing_years:
                df_year = df[df[year_col] == most_recent_year].copy()
                df_year[year_col] = year
                result_dfs.append(df_year)
            
            # Combine all years
            df = pd.concat(result_dfs, ignore_index=True)
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower() or 'region' in col.lower()][0]
    value_col = [col for col in df.columns if 'panjang' in col.lower() or 'ruas' in col.lower()][0]
    
    # Group by region and year, summing the values
    df_grouped = df.groupby([region_col, year_col])[value_col].sum().reset_index()
    
    # Rename columns
    df_result = df_grouped.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'panjang_ruas_jalan'
    })
    df_result['year'] = df_result['year'].astype(int)
    
    # Convert values to numeric (handle comma as decimal separator)
    df_result['panjang_ruas_jalan'] = df_result['panjang_ruas_jalan'].astype(str).str.replace(',', '.').astype(float)
    
    return df_result

def preprocess_sanitasi_layak(df):
    """Preprocess sanitasi_layak data"""
    return preprocess_yearly_data(df, "sanitasi_layak")

def preprocess_titik_layanan_internet(df):
    """Preprocess titik_layanan_internet data"""
    return preprocess_standard_data(df, "titik_layanan_internet")

# Economic indicators preprocessing functions
def preprocess_daftar_upah_minimum(df):
    """
    Preprocess daftar_upah_minimum data
    """
    # Filter for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    value_col = [col for col in df.columns if 'besaran' in col.lower() or 'upah' in col.lower()][0]
    
    # Filter data for required years
    df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
    
    # Check if data exists for all required years
    available_years = df_filtered[year_col].astype(str).unique()
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in daftar_upah_minimum data")
    
    # Rename columns
    df_filtered = df_filtered.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'daftar_upah_minimum'
    })
    
    # Select only required columns
    result_columns = ['wilayah', 'daftar_upah_minimum', 'year']
    df_result = df_filtered[result_columns].copy()
    
    # Convert values to numeric
    df_result['daftar_upah_minimum'] = pd.to_numeric(df_result['daftar_upah_minimum'], errors='coerce')
    
    return df_result

def preprocess_jml_penduduk_bekerja(df):
    """
    Preprocess jml_penduduk_bekerja data
    """
    # Filter for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    value_col = [col for col in df.columns if 'jumlah' in col.lower() or 'penduduk' in col.lower()][0]
    
    # Filter data for required years
    df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
    
    # Check if data exists for all required years
    available_years = df_filtered[year_col].astype(str).unique()
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in jml_penduduk_bekerja data")
    
    # Rename columns
    df_filtered = df_filtered.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'jml_penduduk_bekerja'
    })
    
    # Select only required columns
    result_columns = ['wilayah', 'jml_penduduk_bekerja', 'year']
    df_result = df_filtered[result_columns].copy()
    
    # Convert values to numeric
    df_result['jml_penduduk_bekerja'] = pd.to_numeric(df_result['jml_penduduk_bekerja'], errors='coerce')
    
    return df_result

def preprocess_jml_pengeluaran_per_kapita(df):
    """Preprocess jml_pengeluaran_per_kapita data"""
    return preprocess_standard_data(df, "jml_pengeluaran_per_kapita")

def preprocess_indeks_pembangunan_manusia(df):
    """Preprocess indeks_pembangunan_manusia data"""
    return preprocess_yearly_data(df, "indeks_pembangunan_manusia")

def preprocess_pdrb_harga_konstan(df):
    """Preprocess pdrb_harga_konstan data"""
    return preprocess_yearly_data(df, "pdrb_harga_konstan")

def preprocess_penduduk_miskin(df):
    """Preprocess penduduk_miskin data"""
    return preprocess_yearly_data(df, "penduduk_miskin")

def preprocess_tingkat_pengangguran_terbuka(df):
    """Preprocess tingkat_pengangguran_terbuka data"""
    return preprocess_yearly_data(df, "tingkat_pengangguran_terbuka")

# Health indicators preprocessing functions
def preprocess_angka_harapan_hidup(df):
    """
    Preprocess angka_harapan_hidup data
    """
    # Filter for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']

    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    value_col = [col for col in df.columns if 'angka' in col.lower() or 'harapan' in col.lower()][0]

    # Filter data for required years
    # Clean the year column first, split at comma and take the first part, then remove any decimal point
    df[year_col] = df[year_col].astype(str).str.split(',').str[0].str.split('.').str[0]
    df_filtered = df[df[year_col].isin(years)].copy()

    # Check if data exists for all required years
    available_years = df_filtered[year_col].unique()
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in angka_harapan_hidup data")

    # Rename columns
    df_filtered = df_filtered.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'angka_harapan_hidup'
    })

    # Select only required columns
    result_columns = ['wilayah', 'angka_harapan_hidup', 'year']
    df_result = df_filtered[result_columns].copy()
    df_result['year'] = df_result['year'].astype(int)

    # Convert values to numeric (handle comma as decimal separator)
    df_result['angka_harapan_hidup'] = df_result['angka_harapan_hidup'].astype(str).str.replace(',', '.').astype(float)

    return df_result

def preprocess_fasilitas_kesehatan(df):
    """
    Preprocess fasilitas_kesehatan data, summing different jenis_faskes for the same region
    """
    # Filter data for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    value_col = [col for col in df.columns if 'jumlah' in col.lower()][0]
    
    # Group by region and year, summing the values
    df_grouped = df_filtered.groupby([region_col, year_col])[value_col].sum().reset_index()
    
    # Rename columns
    df_result = df_grouped.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'fasilitas_kesehatan'
    })
    df_result['year'] = df_result['year'].astype(int)
    
    # Convert values to numeric
    df_result['fasilitas_kesehatan'] = pd.to_numeric(df_result['fasilitas_kesehatan'], errors='coerce')
    
    return df_result

def preprocess_kematian_balita(df):
    """Preprocess kematian_balita data"""
    return preprocess_standard_data(df, "kematian_balita")

def preprocess_kematian_bayi(df):
    """Preprocess kematian_bayi data"""
    return preprocess_standard_data(df, "kematian_bayi")

def preprocess_kematian_ibu(df):
    """
    Preprocess kematian_ibu data
    """
    # Filter for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']

    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    value_col = [col for col in df.columns if 'jumlah' in col.lower()][0]

    # Filter data for required years
    # Clean the year column first, split at comma and take the first part, then remove any decimal point
    df[year_col] = df[year_col].astype(str).str.split(',').str[0].str.split('.').str[0]
    df_filtered = df[df[year_col].isin(years)].copy()

    # Check if data exists for all required years
    available_years = df_filtered[year_col].unique()
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in kematian_ibu data")

    # Rename columns
    df_filtered = df_filtered.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'kematian_ibu'
    })

    # Select only required columns
    result_columns = ['wilayah', 'kematian_ibu', 'year']
    df_result = df_filtered[result_columns].copy()
    df_result['year'] = df_result['year'].astype(int)

    # Convert values to numeric (handle comma as decimal separator)
    df_result['kematian_ibu'] = df_result['kematian_ibu'].astype(str).str.replace(',', '.').astype(float)

    return df_result

def preprocess_persentase_balita_stunting(df):
    """
    Preprocess persentase_balita_stunting data
    """
    # Filter for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']

    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    value_col = [col for col in df.columns if 'persentase' in col.lower()][0]

    # Filter data for required years
    # Clean the year column first, split at comma and take the first part, then remove any decimal point
    df[year_col] = df[year_col].astype(str).str.split(',').str[0].str.split('.').str[0]
    df_filtered = df[df[year_col].isin(years)].copy()

    # Check if data exists for all required years
    available_years = df_filtered[year_col].unique()
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in persentase_balita_stunting data")

    # Rename columns
    df_filtered = df_filtered.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'persentase_balita_stunting'
    })

    # Select only required columns
    result_columns = ['wilayah', 'persentase_balita_stunting', 'year']
    df_result = df_filtered[result_columns].copy()
    df_result['year'] = df_result['year'].astype(int)

    # Convert values to numeric (handle comma as decimal separator)
    df_result['persentase_balita_stunting'] = df_result['persentase_balita_stunting'].astype(str).str.replace(',', '.').astype(float)

    return df_result

def preprocess_imunisasi_dasar(df):
    """
    Preprocess imunisasi_dasar data, summing different jenis_kelamin for the same region
    """
    # Filter data for years 2019-2023
    years = ['2019', '2020', '2021', '2022', '2023']
    year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
    df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    value_col = [col for col in df.columns if 'jumlah' in col.lower()][0]
    
    # Group by region and year, summing the values
    df_grouped = df_filtered.groupby([region_col, year_col])[value_col].sum().reset_index()
    
    # Rename columns
    df_result = df_grouped.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: 'imunisasi_dasar'
    })
    df_result['year'] = df_result['year'].astype(int)
    
    # Convert values to numeric
    df_result['imunisasi_dasar'] = pd.to_numeric(df_result['imunisasi_dasar'], errors='coerce')
    
    return df_result

# Education indicators preprocessing functions
def handle_special_case_amh(df):
    """
    Handle special case for angka_melek_huruf where 2019 data is replaced with 2013 data
    """
    # Check if 2013 data exists
    check = df.loc[df['tahun'] == 2013]
    if not check.empty:
        df.loc[df['tahun'] == 2013, 'tahun'] = 2019
        print("Used 2013 data for 2019 in angka_melek_huruf")
    else:
        raise ValueError("2013 data not found in angka_melek_huruf")
    return df

def handle_special_case_apm_apk(df):
    """
    Handle special case for angka_partisipasi_murni and angka_partisipasi_kasar
    where 2023 data is the average of 2021 and 2022 data
    """
    # Check if 2021 and 2022 data exists
    if '2021' in df.columns and '2022' in df.columns:
        # Calculate average of 2021 and 2022 for 2023
        df['2023'] = df[['2021', '2022']].mean(axis=1)
        print("Used average of 2021 and 2022 data for 2023")
    return df

def preprocess_angka_melek_huruf(df):
    """
    Preprocess angka_melek_huruf data with special case handling
    """
    # Handle special case: use 2013 data for 2019
    df = handle_special_case_amh(df)
    return preprocess_standard_data(df, "angka_melek_huruf")

def preprocess_angka_partisipasi_kasar(df):
    """
    Preprocess angka_partisipasi_kasar data, splitting by education level
    """
    # Handle special case: use average of 2021 and 2022 for 2023
    df = handle_special_case_apm_apk(df)
    
    # Identify education levels
    education_levels = df['Pendidikan'].unique()
    
    # Process each education level
    result_dfs = {}
    for level in education_levels:
        # Filter data for this education level
        df_level = df[df['Pendidikan'] == level].copy()
        
        # Create indicator name based on education level
        level_code = level.lower().replace('/', '_').replace(' ', '_')
        indicator_name = f"angka_partisipasi_kasar_{level_code}"
        
        # Process the data
        df_processed = preprocess_yearly_data(df_level, indicator_name)
        
        # Store the result
        result_dfs[indicator_name] = df_processed
    
    return result_dfs

def preprocess_angka_partisipasi_murni(df):
    """
    Preprocess angka_partisipasi_murni data, splitting by education level
    """
    # Handle special case: use average of 2021 and 2022 for 2023
    df = handle_special_case_apm_apk(df)
    
    # Identify education levels
    education_levels = df['Pendidikan'].unique()
    
    # Process each education level
    result_dfs = {}
    for level in education_levels:
        # Filter data for this education level
        df_level = df[df['Pendidikan'] == level].copy()
        
        # Create indicator name based on education level
        level_code = level.lower().replace('/', '_').replace(' ', '_')
        indicator_name = f"angka_partisipasi_murni_{level_code}"
        
        # Process the data
        df_processed = preprocess_yearly_data(df_level, indicator_name)
        
        # Store the result
        result_dfs[indicator_name] = df_processed
    
    return result_dfs

def preprocess_rata_rata_lama_sekolah(df):
    """Preprocess rata_rata_lama_sekolah data"""
    return preprocess_standard_data(df, "rata_rata_lama_sekolah")

# Dictionary mapping indicators to their preprocessing functions
INDICATOR_PROCESSORS = {
    # Infrastructure indicators
    'akses_air_minum': preprocess_akses_air_minum,
    'hunian_layak': preprocess_hunian_layak,
    'kawasan_pariwisata': preprocess_kawasan_pariwisata,
    'kendaraan': preprocess_kendaraan,
    'panjang_ruas_jalan': preprocess_panjang_ruas_jalan,
    'sanitasi_layak': preprocess_sanitasi_layak,
    'titik_layanan_internet': preprocess_titik_layanan_internet,
    
    # Economic indicators
    'daftar_upah_minimum': preprocess_daftar_upah_minimum,
    'jml_penduduk_bekerja': preprocess_jml_penduduk_bekerja,
    'jml_pengeluaran_per_kapita': preprocess_jml_pengeluaran_per_kapita,
    'indeks_pembangunan_manusia': preprocess_indeks_pembangunan_manusia,
    'pdrb_harga_konstan': preprocess_pdrb_harga_konstan,
    'penduduk_miskin': preprocess_penduduk_miskin,
    'tingkat_pengangguran_terbuka': preprocess_tingkat_pengangguran_terbuka,
    
    # Health indicators
    'angka_harapan_hidup': preprocess_angka_harapan_hidup,
    'fasilitas_kesehatan': preprocess_fasilitas_kesehatan,
    'kematian_balita': preprocess_kematian_balita,
    'kematian_bayi': preprocess_kematian_bayi,
    'kematian_ibu': preprocess_kematian_ibu,
    'persentase_balita_stunting': preprocess_persentase_balita_stunting,
    'imunisasi_dasar': preprocess_imunisasi_dasar,
    
    # Education indicators
    'angka_melek_huruf': preprocess_angka_melek_huruf,
    'angka_partisipasi_kasar': preprocess_angka_partisipasi_kasar,
    'angka_partisipasi_murni': preprocess_angka_partisipasi_murni,
    'rata_rata_lama_sekolah': preprocess_rata_rata_lama_sekolah
} 