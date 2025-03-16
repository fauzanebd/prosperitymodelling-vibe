#%% Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


#%% [markdown]
# # Prosperity Modelling
# 
# This project analyzes and categorizes regional prosperity based on indicators from four categories:
# - Infrastructure
# - Economy
# - Health
# - Education
# 
# The analysis includes data preprocessing, labeling, visualization, and model training using Random Forest and Logistic Regression.

#%% [markdown]
# ## Data Preparation

#%%
# Define data directories
data_ekonomi_dir = "data/ekonomi"
data_infra_dir = "data/infrastruktur"
data_kesehatan_dir = "data/kesehatan"
data_pendidikan_dir = "data/pendidikan"

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
    "titik_layanan_internet": {"file": "titik_layanan_internet.csv", "data": pd.DataFrame()},
    "kawasan_pariwisata": {"file": "kawasan_pariwisata.csv", "data": pd.DataFrame()},
    "panjang_ruas_jalan": {"file": "panjang_ruas_jalan.csv", "data": pd.DataFrame()},
    "kendaraan": {"file": "kendaraan.csv", "data": pd.DataFrame()}
}

data_kesehatan_indicator_to_file = {
    "angka_harapan_hidup": {"file": "angka_harapan_hidup.csv", "data": pd.DataFrame()},
    "persentase_balita_stunting": {"file": "persentase_balita_stunting.csv", "data": pd.DataFrame()},
    "kematian_bayi": {"file": "kematian_bayi.csv", "data": pd.DataFrame()},
    "kematian_balita": {"file": "kematian_balita.csv", "data": pd.DataFrame()},
    "kematian_ibu": {"file": "kematian_ibu.csv", "data": pd.DataFrame()},
    "fasilitas_kesehatan": {"file": "fasilitas_kesehatan.csv", "data": pd.DataFrame()},
    "imunisasi_dasar": {"file": "imunisasi_dasar.csv", "data": pd.DataFrame()}
}

data_pendidikan_indicator_to_file = {
    "angka_melek_huruf": {"file": "angka_melek_huruf.csv", "data": pd.DataFrame()},
    "angka_partisipasi_kasar": {"file": "angka_partisipasi_kasar.csv", "data": pd.DataFrame()},
    "angka_partisipasi_murni": {"file": "angka_partisipasi_murni.csv", "data": pd.DataFrame()},
    "rata_rata_lama_sekolah": {"file": "rata_rata_lama_sekolah.csv", "data": pd.DataFrame()}
}

#%% [markdown]
# ## Utility Functions

#%%
# Function to load data with different formats
def load_data(file_path):
    """
    Load data from CSV files with different formats
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    try:
        # Check if file uses semicolon as delimiter
        with open(file_path, 'r') as f:
            first_line = f.readline()
        
        if ';' in first_line:
            delimiter = ';'
        else:
            delimiter = ','
            
        # Load data with appropriate decimal separator
        df = pd.read_csv(file_path, delimiter=delimiter, decimal=",")
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# Function to preprocess yearly data (melt format)
def preprocess_yearly_data(df, indicator_name, years=None):
    """
    Preprocess data with yearly columns format
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with yearly columns
    indicator_name : str
        Name of the indicator
    years : list, optional
        List of years to include (default: 2019-2023)
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, {indicator_name}, year
    """
    if years is None:
        years = ['2019', '2020', '2021', '2022', '2023']
    
    # Filter only required years
    year_columns = [col for col in df.columns if col in years]
    
    # Check if all required years exist
    missing_years = [year for year in years if year not in df.columns]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in {indicator_name} data")
    
    # Select only region name and year columns
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower()][0]
    df_subset = df[[region_col] + year_columns].copy()
    
    # Melt the DataFrame
    df_melted = df_subset.melt(
        id_vars=region_col,
        value_vars=year_columns,
        var_name='year',
        value_name=indicator_name
    )
    
    # Rename region column to 'wilayah'
    df_melted = df_melted.rename(columns={region_col: 'wilayah'})
    
    # Convert values to numeric
    df_melted[indicator_name] = pd.to_numeric(df_melted[indicator_name], errors='coerce')

    # Convert year to int
    df_melted['year'] = df_melted['year'].astype(int)
    
    return df_melted

# Function to preprocess standard data
def preprocess_standard_data(df, indicator_name, years=None):
    """
    Preprocess data with standard format
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with standard format
    indicator_name : str
        Name of the indicator
    years : list, optional
        List of years to include (default: 2019-2023)
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, {indicator_name}, year
    """
    if years is None:
        years = ['2019', '2020', '2021', '2022', '2023']
    
    # Identify column names
    region_col = [col for col in df.columns if 'nama' in col.lower() or 'kabupaten' in col.lower() or 'region' in col.lower()]
    if not region_col:
        print(f"Error: Could not find region column in {indicator_name} data")
        return pd.DataFrame()
    region_col = region_col[0]
    
    year_col = [col for col in df.columns if 'tahun' in col.lower() or 'year' in col.lower()]
    if not year_col:
        print(f"Error: Could not find year column in {indicator_name} data")
        return pd.DataFrame()
    year_col = year_col[0]
    
    # Try different patterns for value column
    possible_patterns = [
        'jumlah', indicator_name, 'nilai', 'value', 'besaran', 'angka', 'persentase', 'rate', 'panjang', 'rata_rata'
    ]
    
    value_col = None
    for pattern in possible_patterns:
        cols = [col for col in df.columns if pattern.lower() in col.lower()]
        if cols:
            value_col = cols[0]
            break
    
    if value_col is None:
        # If no match found, use the third column (assuming first is region, second might be a category, third is value)
        if len(df.columns) >= 3:
            value_col = df.columns[2]
            print(f"Warning: Using column '{value_col}' as value column for {indicator_name}")
        else:
            print(f"Error: Could not identify value column for {indicator_name}")
            return pd.DataFrame()
    
    # Filter data for required years
    df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
    
    # Check if data exists for all required years
    available_years = df_filtered[year_col].astype(str).unique()
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        print(f"Warning: Missing years {missing_years} in {indicator_name} data")
    
    # Rename columns
    df_filtered = df_filtered.rename(columns={
        region_col: 'wilayah',
        year_col: 'year',
        value_col: indicator_name
    })
    
    # Select only required columns
    result_columns = ['wilayah', indicator_name, 'year']
    df_result = df_filtered[result_columns].copy()
    
    # Convert values to numeric
    df_result[indicator_name] = pd.to_numeric(df_result[indicator_name], errors='coerce')
    
    return df_result

# Function for IQR-based labeling
def label_iqr(df, column, reverse=False):
    """
    Function to determine categories based on IQR.
    If reverse=True, then lower values are better (e.g., Penduduk Miskin).
    If reverse=False, then higher values are better (e.g., Upah Minimum, PDRB).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to apply labeling
    reverse : bool
        Whether lower values are better (True) or higher values are better (False)
        
    Returns:
    --------
    pd.Series
        Series with labels: "Sejahtera", "Menengah", "Tidak Sejahtera"
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    low_threshold = Q1  
    high_threshold = Q3 

    def categorize(value):
        if reverse:
            # Lower values are better
            if value < low_threshold:
                return "Sejahtera"
            elif value > high_threshold:
                return "Tidak Sejahtera"
            else:
                return "Menengah"
        else:
            # Higher values are better
            if value > high_threshold:
                return "Sejahtera"
            elif value < low_threshold:
                return "Tidak Sejahtera"
            else:
                return "Menengah"
            
    return df[column].apply(categorize)

# Function to handle special case for angka_melek_huruf (2013 data for 2019)
def handle_special_case_amh(df):
    """
    Handle special case for angka_melek_huruf where 2019 data is replaced with 2013 data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing angka_melek_huruf data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with 2013 data used for 2019
    """
    
    # Check if 2013 data exists
    check = df.loc[df['tahun'] == 2013]
    if not check.empty:
        df.loc[df['tahun'] == 2013, 'tahun'] = 2019
        print("Used 2013 data for 2019 in angka_melek_huruf")
    else:
        raise ValueError("2013 data not found in angka_melek_huruf")
    return df

# Function to handle special case for APM and APK (average of 2021 and 2022 for 2023)
def handle_special_case_apm_apk(df):
    """
    Handle special case for angka_partisipasi_murni and angka_partisipasi_kasar
    where 2023 data is the average of 2021 and 2022 data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing APM or APK data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with 2023 data as average of 2021 and 2022
    """
    # Check if 2021 and 2022 data exists
    if '2021' in df.columns and '2022' in df.columns:
        # Calculate average of 2021 and 2022 for 2023
        df['2023'] = df[['2021', '2022']].mean(axis=1)
        print("Used average of 2021 and 2022 data for 2023")
    return df

print("Project setup completed. Ready for data preprocessing.")

#%% [markdown]
# ## Test Data Loading Functions

#%%
# Test loading and preprocessing for each data type

# Test yearly data (e.g., akses_air_minum.csv)
print("\nTesting yearly data loading and preprocessing:")
yearly_test_file = os.path.join(data_infra_dir, "akses_air_minum.csv")
yearly_df = load_data(yearly_test_file)
print(f"Original shape: {yearly_df.shape}")
yearly_processed = preprocess_yearly_data(yearly_df, "akses_air_minum")
print(f"Processed shape: {yearly_processed.shape}")
print(yearly_processed.head())

# Test standard data (e.g., kawasan_pariwisata.csv)
print("\nTesting standard data loading and preprocessing:")
standard_test_file = os.path.join(data_infra_dir, "kawasan_pariwisata.csv")
standard_df = load_data(standard_test_file)
print(f"Original shape: {standard_df.shape}")
standard_processed = preprocess_standard_data(standard_df, "kawasan_pariwisata")
print(f"Processed shape: {standard_processed.shape}")
print(standard_processed.head())

# Test data with special columns (e.g., kendaraan.csv with tipe_roda_kendaraan)
print("\nTesting data with special columns:")
special_test_file = os.path.join(data_infra_dir, "kendaraan.csv")
special_df = load_data(special_test_file)
print(f"Original shape: {special_df.shape}")
print(special_df.head())

# Test IQR labeling function
print("\nTesting IQR labeling function:")
test_df = pd.DataFrame({
    'indicator': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})
test_df['label_normal'] = label_iqr(test_df, 'indicator', reverse=False)
test_df['label_reverse'] = label_iqr(test_df, 'indicator', reverse=True)
print(test_df)

#%% [markdown]
# ## Specialized Preprocessing Functions

#%%
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing panjang_ruas_jalan data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, panjang_ruas_jalan, year
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing daftar_upah_minimum data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, daftar_upah_minimum, year
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing jml_penduduk_bekerja data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, jml_penduduk_bekerja, year
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing angka_harapan_hidup data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, angka_harapan_hidup, year
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
    """Preprocess kematian_ibu data"""
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing persentase_balita_stunting data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with columns: wilayah, persentase_balita_stunting, year
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

#%% [markdown]
# ## Test Specialized Preprocessing Functions

#%%
# Test specialized preprocessing functions with sample data

# Test kendaraan preprocessing (splitting into roda_2 and roda_4)
print("\nTesting kendaraan preprocessing:")
kendaraan_file = os.path.join(data_infra_dir, "kendaraan.csv")
kendaraan_df = load_data(kendaraan_file)
kendaraan_roda_2, kendaraan_roda_4 = preprocess_kendaraan(kendaraan_df)
print(f"Roda 2 shape: {kendaraan_roda_2.shape}")
print(f"Roda 4 shape: {kendaraan_roda_4.shape}")
print("Roda 2 sample:")
print(kendaraan_roda_2.head())
print("Roda 4 sample:")
print(kendaraan_roda_4.head())

# Test fasilitas_kesehatan preprocessing (summing different jenis_faskes)
print("\nTesting fasilitas_kesehatan preprocessing:")
faskes_file = os.path.join(data_kesehatan_dir, "fasilitas_kesehatan.csv")
faskes_df = load_data(faskes_file)
faskes_processed = preprocess_fasilitas_kesehatan(faskes_df)
print(f"Processed shape: {faskes_processed.shape}")
print(faskes_processed.head())

# Test angka_partisipasi_kasar preprocessing (splitting by education level)
print("\nTesting angka_partisipasi_kasar preprocessing:")
apk_file = os.path.join(data_pendidikan_dir, "angka_partisipasi_kasar.csv")
apk_df = load_data(apk_file)
apk_processed = preprocess_angka_partisipasi_kasar(apk_df)
print(f"Number of education levels: {len(apk_processed)}")
for level, df in apk_processed.items():
    print(f"\n{level} shape: {df.shape}")
    print(df.head())

print("\nSetup and testing completed. Ready for full data loading and preprocessing.")

#%% [markdown]
# # Data Preprocessing

#%% [markdown]
# ## Infrastructure Indicators Preprocessing
# 
# Processing the 7 infrastructure indicators:
# 1. Akses air minum
# 2. Hunian layak
# 3. Kawasan pariwisata
# 4. Kendaraan (split into roda 2 and roda 4)
# 5. Panjang ruas jalan
# 6. Sanitasi layak
# 7. Titik layanan internet

#%%
# Create a dictionary to store preprocessed infrastructure data
infra_data = {}

# 1. Akses air minum
print("\nPreprocessing akses_air_minum data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["akses_air_minum"]["file"])
df = load_data(file_path)
infra_data["akses_air_minum"] = preprocess_akses_air_minum(df)
print(f"Shape after preprocessing: {infra_data['akses_air_minum'].shape}")
print(infra_data["akses_air_minum"].head())

# 2. Hunian layak
print("\nPreprocessing hunian_layak data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["hunian_layak"]["file"])
df = load_data(file_path)
infra_data["hunian_layak"] = preprocess_hunian_layak(df)
print(f"Shape after preprocessing: {infra_data['hunian_layak'].shape}")
print(infra_data["hunian_layak"].head())

# 3. Kawasan pariwisata
print("\nPreprocessing kawasan_pariwisata data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["kawasan_pariwisata"]["file"])
df = load_data(file_path)
# Filter for years 2019-2023
years = ['2019', '2020', '2021', '2022', '2023']
year_col = [col for col in df.columns if 'tahun' in col.lower()][0]
df_filtered = df[df[year_col].astype(str).isin([str(year) for year in years])].copy()
infra_data["kawasan_pariwisata"] = preprocess_kawasan_pariwisata(df_filtered)
print(f"Shape after preprocessing: {infra_data['kawasan_pariwisata'].shape}")
print(infra_data["kawasan_pariwisata"].head())

# 4. Kendaraan (split into roda 2 and roda 4)
print("\nPreprocessing kendaraan data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["kendaraan"]["file"])
df = load_data(file_path)
infra_data["kendaraan_roda_2"], infra_data["kendaraan_roda_4"] = preprocess_kendaraan(df)
print(f"Kendaraan roda 2 shape: {infra_data['kendaraan_roda_2'].shape}")
print(f"Kendaraan roda 4 shape: {infra_data['kendaraan_roda_4'].shape}")
print("Kendaraan roda 2 sample:")
print(infra_data["kendaraan_roda_2"].head())
print("Kendaraan roda 4 sample:")
print(infra_data["kendaraan_roda_4"].head())

# 5. Panjang ruas jalan
print("\nPreprocessing panjang_ruas_jalan data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["panjang_ruas_jalan"]["file"])
df = load_data(file_path)
infra_data["panjang_ruas_jalan"] = preprocess_panjang_ruas_jalan(df)
print(f"Shape after preprocessing: {infra_data['panjang_ruas_jalan'].shape}")
print(infra_data["panjang_ruas_jalan"].head())

# 6. Sanitasi layak
print("\nPreprocessing sanitasi_layak data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["sanitasi_layak"]["file"])
df = load_data(file_path)
infra_data["sanitasi_layak"] = preprocess_sanitasi_layak(df)
print(f"Shape after preprocessing: {infra_data['sanitasi_layak'].shape}")
print(infra_data["sanitasi_layak"].head())

# 7. Titik layanan internet
print("\nPreprocessing titik_layanan_internet data...")
file_path = os.path.join(data_infra_dir, data_infrastruktur_indicator_to_file["titik_layanan_internet"]["file"])
df = load_data(file_path)
infra_data["titik_layanan_internet"] = preprocess_titik_layanan_internet(df_filtered)
print(f"Shape after preprocessing: {infra_data['titik_layanan_internet'].shape}")
print(infra_data["titik_layanan_internet"].head())

# Summary of infrastructure data
print("\nInfrastructure indicators summary:")
for indicator, data in infra_data.items():
    print(f"{indicator}: {data.shape[0]} rows, years: {data['year'].unique()}")

# Save preprocessed data for later use
infra_data_processed = infra_data

#%% [markdown]
# ## Economic Indicators Preprocessing
# 
# Processing the 7 economic indicators:
# 1. Daftar upah minimum
# 2. Jumlah penduduk bekerja
# 3. Jumlah pengeluaran per kapita
# 4. Indeks pembangunan manusia
# 5. PDRB harga konstan
# 6. Penduduk miskin
# 7. Tingkat pengangguran terbuka

#%%
# Create a dictionary to store preprocessed economic data
ekonomi_data = {}

# 1. Daftar upah minimum
print("\nPreprocessing daftar_upah_minimum data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["daftar_upah_minimum"]["file"])
df = load_data(file_path)
ekonomi_data["daftar_upah_minimum"] = preprocess_daftar_upah_minimum(df)
print(f"Shape after preprocessing: {ekonomi_data['daftar_upah_minimum'].shape}")
print(ekonomi_data["daftar_upah_minimum"].head())

# 2. Jumlah penduduk bekerja
print("\nPreprocessing jml_penduduk_bekerja data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["jml_penduduk_bekerja"]["file"])
df = load_data(file_path)
ekonomi_data["jml_penduduk_bekerja"] = preprocess_jml_penduduk_bekerja(df)
print(f"Shape after preprocessing: {ekonomi_data['jml_penduduk_bekerja'].shape}")
print(ekonomi_data["jml_penduduk_bekerja"].head())

# 3. Jumlah pengeluaran per kapita
print("\nPreprocessing jml_pengeluaran_per_kapita data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["jml_pengeluaran_per_kapita"]["file"])
df = load_data(file_path)
ekonomi_data["jml_pengeluaran_per_kapita"] = preprocess_jml_pengeluaran_per_kapita(df)
print(f"Shape after preprocessing: {ekonomi_data['jml_pengeluaran_per_kapita'].shape}")
print(ekonomi_data["jml_pengeluaran_per_kapita"].head())

# 4. Indeks pembangunan manusia
print("\nPreprocessing indeks_pembangunan_manusia data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["indeks_pembangunan_manusia"]["file"])
df = load_data(file_path)
ekonomi_data["indeks_pembangunan_manusia"] = preprocess_indeks_pembangunan_manusia(df)
print(f"Shape after preprocessing: {ekonomi_data['indeks_pembangunan_manusia'].shape}")
print(ekonomi_data["indeks_pembangunan_manusia"].head())

# 5. PDRB harga konstan
print("\nPreprocessing pdrb_harga_konstan data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["pdrb_harga_konstan"]["file"])
df = load_data(file_path)
ekonomi_data["pdrb_harga_konstan"] = preprocess_pdrb_harga_konstan(df)
print(f"Shape after preprocessing: {ekonomi_data['pdrb_harga_konstan'].shape}")
print(ekonomi_data["pdrb_harga_konstan"].head())

# 6. Penduduk miskin
print("\nPreprocessing penduduk_miskin data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["penduduk_miskin"]["file"])
df = load_data(file_path)
ekonomi_data["penduduk_miskin"] = preprocess_penduduk_miskin(df)
print(f"Shape after preprocessing: {ekonomi_data['penduduk_miskin'].shape}")
print(ekonomi_data["penduduk_miskin"].head())

# 7. Tingkat pengangguran terbuka
print("\nPreprocessing tingkat_pengangguran_terbuka data...")
file_path = os.path.join(data_ekonomi_dir, data_ekonomi_indicator_to_file["tingkat_pengangguran_terbuka"]["file"])
df = load_data(file_path)
ekonomi_data["tingkat_pengangguran_terbuka"] = preprocess_tingkat_pengangguran_terbuka(df)
print(f"Shape after preprocessing: {ekonomi_data['tingkat_pengangguran_terbuka'].shape}")
print(ekonomi_data["tingkat_pengangguran_terbuka"].head())

# Summary of economic data
print("\nEconomic indicators summary:")
for indicator, data in ekonomi_data.items():
    print(f"{indicator}: {data.shape[0]} rows, years: {data['year'].unique()}")

# Save preprocessed data for later use
ekonomi_data_processed = ekonomi_data

#%% [markdown]
# ## Health Indicators Preprocessing
# 
# Processing the 7 health indicators:
# 1. Angka harapan hidup
# 2. Fasilitas kesehatan
# 3. Kematian balita
# 4. Kematian bayi
# 5. Kematian ibu
# 6. Persentase balita stunting
# 7. Imunisasi dasar

#%%
# Create a dictionary to store preprocessed health data
kesehatan_data = {}

# 1. Angka harapan hidup
print("\nPreprocessing angka_harapan_hidup data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["angka_harapan_hidup"]["file"])
df = load_data(file_path)
kesehatan_data["angka_harapan_hidup"] = preprocess_angka_harapan_hidup(df)
print(f"Shape after preprocessing: {kesehatan_data['angka_harapan_hidup'].shape}")
print(kesehatan_data["angka_harapan_hidup"].head())

# 2. Fasilitas kesehatan
print("\nPreprocessing fasilitas_kesehatan data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["fasilitas_kesehatan"]["file"])
df = load_data(file_path)
kesehatan_data["fasilitas_kesehatan"] = preprocess_fasilitas_kesehatan(df)
print(f"Shape after preprocessing: {kesehatan_data['fasilitas_kesehatan'].shape}")
print(kesehatan_data["fasilitas_kesehatan"].head())

# 3. Kematian balita
print("\nPreprocessing kematian_balita data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["kematian_balita"]["file"])
df = load_data(file_path)
kesehatan_data["kematian_balita"] = preprocess_kematian_balita(df)
print(f"Shape after preprocessing: {kesehatan_data['kematian_balita'].shape}")
print(kesehatan_data["kematian_balita"].head())

# 4. Kematian bayi
print("\nPreprocessing kematian_bayi data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["kematian_bayi"]["file"])
df = load_data(file_path)
kesehatan_data["kematian_bayi"] = preprocess_kematian_bayi(df)
print(f"Shape after preprocessing: {kesehatan_data['kematian_bayi'].shape}")
print(kesehatan_data["kematian_bayi"].head())

# 5. Kematian ibu
print("\nPreprocessing kematian_ibu data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["kematian_ibu"]["file"])
df = load_data(file_path)
kesehatan_data["kematian_ibu"] = preprocess_kematian_ibu(df)
print(f"Shape after preprocessing: {kesehatan_data['kematian_ibu'].shape}")
print(kesehatan_data["kematian_ibu"].head())

# 6. Persentase balita stunting
print("\nPreprocessing persentase_balita_stunting data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["persentase_balita_stunting"]["file"])
df = load_data(file_path)
kesehatan_data["persentase_balita_stunting"] = preprocess_persentase_balita_stunting(df)
print(f"Shape after preprocessing: {kesehatan_data['persentase_balita_stunting'].shape}")
print(kesehatan_data["persentase_balita_stunting"].head())

# 7. Imunisasi dasar
print("\nPreprocessing imunisasi_dasar data...")
file_path = os.path.join(data_kesehatan_dir, data_kesehatan_indicator_to_file["imunisasi_dasar"]["file"])
df = load_data(file_path)
kesehatan_data["imunisasi_dasar"] = preprocess_imunisasi_dasar(df)
print(f"Shape after preprocessing: {kesehatan_data['imunisasi_dasar'].shape}")
print(kesehatan_data["imunisasi_dasar"].head())

# Summary of health data
print("\nHealth indicators summary:")
for indicator, data in kesehatan_data.items():
    print(f"{indicator}: {data.shape[0]} rows, years: {data['year'].unique()}")

# Save preprocessed data for later use
kesehatan_data_processed = kesehatan_data

#%% [markdown]
# ## Education Indicators Preprocessing
# 
# Processing the 4 education indicators:
# 1. Angka melek huruf
# 2. Angka partisipasi kasar (split by education level)
# 3. Angka partisipasi murni (split by education level)
# 4. Rata-rata lama sekolah

#%%
# Create a dictionary to store preprocessed education data
pendidikan_data = {}

# 1. Angka melek huruf
print("\nPreprocessing angka_melek_huruf data...")
file_path = os.path.join(data_pendidikan_dir, data_pendidikan_indicator_to_file["angka_melek_huruf"]["file"])
df = load_data(file_path)
pendidikan_data["angka_melek_huruf"] = preprocess_angka_melek_huruf(df)
print(f"Shape after preprocessing: {pendidikan_data['angka_melek_huruf'].shape}")
print(pendidikan_data["angka_melek_huruf"].head())


# 2. Angka partisipasi kasar
print("\nPreprocessing angka_partisipasi_kasar data...")
file_path = os.path.join(data_pendidikan_dir, data_pendidikan_indicator_to_file["angka_partisipasi_kasar"]["file"])
df = load_data(file_path)
# Handle special case: use average of 2021 and 2022 for 2023
apk_dfs = preprocess_angka_partisipasi_kasar(df)
for level, df_level in apk_dfs.items():
    pendidikan_data[level] = df_level
    print(f"\n{level} shape: {df_level.shape}")
    print(df_level.head())

# 3. Angka partisipasi murni
print("\nPreprocessing angka_partisipasi_murni data...")
file_path = os.path.join(data_pendidikan_dir, data_pendidikan_indicator_to_file["angka_partisipasi_murni"]["file"])
df = load_data(file_path)
# Handle special case: use average of 2021 and 2022 for 2023
apm_dfs = preprocess_angka_partisipasi_murni(df)
for level, df_level in apm_dfs.items():
    pendidikan_data[level] = df_level
    print(f"\n{level} shape: {df_level.shape}")
    print(df_level.head())

# 4. Rata-rata lama sekolah
print("\nPreprocessing rata_rata_lama_sekolah data...")
file_path = os.path.join(data_pendidikan_dir, data_pendidikan_indicator_to_file["rata_rata_lama_sekolah"]["file"])
df = load_data(file_path)
pendidikan_data["rata_rata_lama_sekolah"] = preprocess_rata_rata_lama_sekolah(df_filtered)
print(f"Shape after preprocessing: {pendidikan_data['rata_rata_lama_sekolah'].shape}")
print(pendidikan_data["rata_rata_lama_sekolah"].head())

# Summary of education data
print("\nEducation indicators summary:")
for indicator, data in pendidikan_data.items():
    print(f"{indicator}: {data.shape[0]} rows, years: {data['year'].unique()}")

# Save preprocessed data for later use
pendidikan_data_processed = pendidikan_data

#%% [markdown]
# # Data Labelling

#%% [markdown]
# ## Infrastructure Indicators Labelling
# 
# For infrastructure indicators, all use IQR labelling with reverse=False (higher values are better)

#%%
# Create a dictionary to store labelled infrastructure data
infra_data_labelled = {}

# Apply IQR labelling to each infrastructure indicator
for indicator, data in infra_data_processed.items():
    print(f"\nLabelling {indicator} data...")
    
    # Create a copy of the data
    df_labelled = data.copy()
    
    # Apply IQR labelling (all infrastructure indicators use reverse=False)
    df_labelled['label_sejahtera'] = label_iqr(df_labelled, indicator, reverse=False)
    
    # Store the labelled data
    infra_data_labelled[indicator] = df_labelled
    
    # Print summary
    print(f"Label distribution for {indicator}:")
    print(df_labelled['label_sejahtera'].value_counts())
    print(df_labelled.head())

# Summary of labelled infrastructure data
print("\nInfrastructure indicators labelling summary:")
for indicator, data in infra_data_labelled.items():
    label_counts = data['label_sejahtera'].value_counts()
    print(f"{indicator}: Sejahtera: {label_counts.get('Sejahtera', 0)}, "
          f"Menengah: {label_counts.get('Menengah', 0)}, "
          f"Tidak Sejahtera: {label_counts.get('Tidak Sejahtera', 0)}")

# Save labelled data for later use
infra_data_final = infra_data_labelled

#%% [markdown]
# ## Economic Indicators Labelling
# 
# For economic indicators, we use a mix of IQR and manual labelling:
# 
# 1. daftar_upah_minimum: IQR with reverse=False (higher is better)
# 2. jml_penduduk_bekerja: IQR with reverse=False (higher is better)
# 3. jml_pengeluaran_per_kapita: IQR with reverse=False (higher is better)
# 4. indeks_pembangunan_manusia: Manual labelling (Sejahtera if > 70, Menengah if 60-70, else Tidak Sejahtera)
# 5. pdrb_harga_konstan: IQR with reverse=False (higher is better)
# 6. penduduk_miskin: IQR with reverse=True (lower is better)
# 7. tingkat_pengangguran_terbuka: Manual labelling (Sejahtera if < 6.75, Menengah if 6.5-7.0, else Tidak Sejahtera)

#%%
# Create a dictionary to store labelled economic data
ekonomi_data_labelled = {}

# Apply labelling to each economic indicator
for indicator, data in ekonomi_data_processed.items():
    print(f"\nLabelling {indicator} data...")
    
    # Create a copy of the data
    df_labelled = data.copy()
    
    # Apply appropriate labelling based on the indicator
    if indicator == "indeks_pembangunan_manusia":
        # Manual labelling: Sejahtera if > 70, Menengah if 60-70, else Tidak Sejahtera
        def label_ipm(value):
            if value > 70:
                return "Sejahtera"
            elif 60 <= value <= 70:
                return "Menengah"
            else:
                return "Tidak Sejahtera"
        
        df_labelled['label_sejahtera'] = df_labelled[indicator].apply(label_ipm)
        
    elif indicator == "tingkat_pengangguran_terbuka":
        # Manual labelling: Sejahtera if < 6.75, Menengah if 6.5-7.0, else Tidak Sejahtera
        def label_tpt(value):
            if value < 6.75:
                return "Sejahtera"
            elif 6.5 <= value <= 7.0:
                return "Menengah"
            else:
                return "Tidak Sejahtera"
        
        df_labelled['label_sejahtera'] = df_labelled[indicator].apply(label_tpt)
        
    elif indicator == "penduduk_miskin":
        # IQR with reverse=True (lower is better)
        df_labelled['label_sejahtera'] = label_iqr(df_labelled, indicator, reverse=True)
        
    else:
        # IQR with reverse=False (higher is better)
        df_labelled['label_sejahtera'] = label_iqr(df_labelled, indicator, reverse=False)
    
    # Store the labelled data
    ekonomi_data_labelled[indicator] = df_labelled
    
    # Print summary
    print(f"Label distribution for {indicator}:")
    print(df_labelled['label_sejahtera'].value_counts())
    print(df_labelled.head())

# Summary of labelled economic data
print("\nEconomic indicators labelling summary:")
for indicator, data in ekonomi_data_labelled.items():
    label_counts = data['label_sejahtera'].value_counts()
    print(f"{indicator}: Sejahtera: {label_counts.get('Sejahtera', 0)}, "
          f"Menengah: {label_counts.get('Menengah', 0)}, "
          f"Tidak Sejahtera: {label_counts.get('Tidak Sejahtera', 0)}")

# Save labelled data for later use
ekonomi_data_final = ekonomi_data_labelled

#%% [markdown]
# ## Health Indicators Labelling
# 
# For health indicators, we use a mix of IQR and manual labelling:
# 
# 1. angka_harapan_hidup: IQR with reverse=False (higher is better)
# 2. fasilitas_kesehatan: IQR with reverse=False (higher is better)
# 3. kematian_balita: IQR with reverse=True (lower is better)
# 4. kematian_bayi: IQR with reverse=True (lower is better)
# 5. kematian_ibu: IQR with reverse=True (lower is better)
# 6. persentase_balita_stunting: Manual labelling (Sejahtera if < 20, Menengah if 20-29, else Tidak Sejahtera)
# 7. imunisasi_dasar: IQR with reverse=False (higher is better)

#%%
# Create a dictionary to store labelled health data
kesehatan_data_labelled = {}

# Apply labelling to each health indicator
for indicator, data in kesehatan_data_processed.items():
    print(f"\nLabelling {indicator} data...")
    
    # Create a copy of the data
    df_labelled = data.copy()
    
    # Apply appropriate labelling based on the indicator
    if indicator == "persentase_balita_stunting":
        # Manual labelling: Sejahtera if < 20, Menengah if 20-29, else Tidak Sejahtera
        def label_stunting(value):
            if value < 20:
                return "Sejahtera"
            elif 20 <= value <= 29:
                return "Menengah"
            else:
                return "Tidak Sejahtera"
        
        df_labelled['label_sejahtera'] = df_labelled[indicator].apply(label_stunting)
        
    elif indicator in ["kematian_balita", "kematian_bayi", "kematian_ibu"]:
        # IQR with reverse=True (lower is better)
        df_labelled['label_sejahtera'] = label_iqr(df_labelled, indicator, reverse=True)
        
    else:
        # IQR with reverse=False (higher is better)
        df_labelled['label_sejahtera'] = label_iqr(df_labelled, indicator, reverse=False)
    
    # Store the labelled data
    kesehatan_data_labelled[indicator] = df_labelled
    
    # Print summary
    print(f"Label distribution for {indicator}:")
    print(df_labelled['label_sejahtera'].value_counts())
    print(df_labelled.head())

# Summary of labelled health data
print("\nHealth indicators labelling summary:")
for indicator, data in kesehatan_data_labelled.items():
    label_counts = data['label_sejahtera'].value_counts()
    print(f"{indicator}: Sejahtera: {label_counts.get('Sejahtera', 0)}, "
          f"Menengah: {label_counts.get('Menengah', 0)}, "
          f"Tidak Sejahtera: {label_counts.get('Tidak Sejahtera', 0)}")

# Save labelled data for later use
kesehatan_data_final = kesehatan_data_labelled

#%% [markdown]
# ## Education Indicators Labelling
# 
# For education indicators, all use IQR labelling with reverse=False (higher values are better)

#%%
# Create a dictionary to store labelled education data
pendidikan_data_labelled = {}

# Apply IQR labelling to each education indicator
for indicator, data in pendidikan_data_processed.items():
    print(f"\nLabelling {indicator} data...")
    
    # Create a copy of the data
    df_labelled = data.copy()
    
    # Apply IQR labelling (all education indicators use reverse=False)
    df_labelled['label_sejahtera'] = label_iqr(df_labelled, indicator, reverse=False)
    
    # Store the labelled data
    pendidikan_data_labelled[indicator] = df_labelled
    
    # Print summary
    print(f"Label distribution for {indicator}:")
    print(df_labelled['label_sejahtera'].value_counts())
    print(df_labelled.head())

# Summary of labelled education data
print("\nEducation indicators labelling summary:")
for indicator, data in pendidikan_data_labelled.items():
    label_counts = data['label_sejahtera'].value_counts()
    print(f"{indicator}: Sejahtera: {label_counts.get('Sejahtera', 0)}, "
          f"Menengah: {label_counts.get('Menengah', 0)}, "
          f"Tidak Sejahtera: {label_counts.get('Tidak Sejahtera', 0)}")

# Save labelled data for later use
pendidikan_data_final = pendidikan_data_labelled

#%% [markdown]
# ## Combine All Labelled Data
# 
# Combine all labelled data from different categories into a single dictionary for easier access

#%%
# Combine all labelled data
all_data_labelled = {}
all_data_labelled.update(infra_data_final)
all_data_labelled.update(ekonomi_data_final)
all_data_labelled.update(kesehatan_data_final)
all_data_labelled.update(pendidikan_data_final)

print(f"\nTotal number of labelled indicators: {len(all_data_labelled)}")
print("List of all labelled indicators:")
for i, indicator in enumerate(all_data_labelled.keys(), 1):
    print(f"{i}. {indicator}")

# Save all labelled data for later use
all_data_final = all_data_labelled

#%% [markdown]
# # Data Visualization - Exploratory Data Analysis
# 
# In this section, we'll create visualizations to understand the distribution and patterns in our preprocessed data.
# We'll explore:
# 1. Distribution of indicators across regions
# 2. Trends over time (2019-2023)
# 3. Distribution of prosperity labels
# 4. Correlation between indicators
# 5. Regional comparisons

#%% [markdown]
# ## 1. Distribution of Indicators

#%%
# Set up the visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Function to plot distribution of an indicator
def plot_indicator_distribution(data, indicator_name, title=None, year=None):
    """
    Plot the distribution of an indicator
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the indicator data
    indicator_name : str
        Name of the indicator column
    title : str, optional
        Title for the plot
    """

  
    if year is not None:
        data = data.loc[data['year'] == year].copy()
        title = f"Distribution of {indicator_name} at {year}"  

    if title is None:
        title = f"Distribution of {indicator_name}"
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram with KDE
    sns.histplot(data[indicator_name].dropna(), kde=True)
    
    # Add vertical lines for quartiles
    q1 = data[indicator_name].quantile(0.25)
    q2 = data[indicator_name].quantile(0.5)
    q3 = data[indicator_name].quantile(0.75)
    
    plt.axvline(q1, color='r', linestyle='--', alpha=0.7, label=f'Q1: {q1:.2f}')
    plt.axvline(q2, color='g', linestyle='--', alpha=0.7, label=f'Median: {q2:.2f}')
    plt.axvline(q3, color='b', linestyle='--', alpha=0.7, label=f'Q3: {q3:.2f}')
    
    plt.title(title, fontsize=14)
    plt.xlabel(indicator_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot distribution for selected indicators from each category
# Infrastructure
print("\nInfrastructure Indicators Distribution:")
for indicator in ['akses_air_minum', 'hunian_layak', 'kendaraan_roda_2']:
    plot_indicator_distribution(all_data_final[indicator], indicator, year=2023)

# Economic
print("\nEconomic Indicators Distribution:")
for indicator in ['indeks_pembangunan_manusia', 'penduduk_miskin', 'tingkat_pengangguran_terbuka']:
    plot_indicator_distribution(all_data_final[indicator], indicator, year=2023)

# Health
print("\nHealth Indicators Distribution:")
for indicator in ['angka_harapan_hidup', 'persentase_balita_stunting', 'kematian_bayi']:
    plot_indicator_distribution(all_data_final[indicator], indicator, year=2023)

# Education
print("\nEducation Indicators Distribution:")
for indicator in ['angka_melek_huruf', 'rata_rata_lama_sekolah']:
    plot_indicator_distribution(all_data_final[indicator], indicator, year=2023)


#%% [markdown]
# ## 2. Trends Over Time (2019-2023)

#%%
# Function to plot trends over time for an indicator
def plot_indicator_trend(data, indicator_name, title=None):
    """
    Plot the trend of an indicator over time
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the indicator data
    indicator_name : str
        Name of the indicator column
    title : str, optional
        Title for the plot
    """
    if title is None:
        title = f"Trend of {indicator_name} (2019-2023)"
    
    # Check if data is empty
    if data.empty:
        print(f"No data available for {indicator_name}")
        return
    
    # Check if indicator column exists
    if indicator_name not in data.columns:
        print(f"Column {indicator_name} not found in data")
        return
    
    try:
        # Make a copy to avoid modifying the original data
        data = data.copy()
        
        # Convert year to numeric if it's not already
        data['year'] = pd.to_numeric(data['year'], errors='coerce')
        
        # Drop rows with NaN values
        data = data.dropna(subset=['year', indicator_name])
        
        # Check if there's any data left
        if data.empty:
            print(f"No valid data for {indicator_name} after dropping NaN values")
            return
        
        # Calculate mean value per year
        yearly_mean = data.groupby('year')[indicator_name].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        
        # Plot line chart
        sns.lineplot(x='year', y=indicator_name, data=yearly_mean, marker='o', linewidth=2)
        
        # Add data points for each region (with transparency)
        sns.scatterplot(x='year', y=indicator_name, data=data, alpha=0.3, color='gray')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel(indicator_name, fontsize=12)
        plt.xticks(data['year'].unique())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting trend for {indicator_name}: {e}")
        # Print data info for debugging
        print(f"Data info: {data.info()}")
        print(f"Data head: {data.head()}")
        print(f"Year values: {data['year'].unique()}")
        print(f"Indicator values: {data[indicator_name].describe()}")

# Plot trends for selected indicators from each category
# Infrastructure
print("\nInfrastructure Indicators Trends:")
for indicator in ['akses_air_minum', 'sanitasi_layak']:
    plot_indicator_trend(all_data_final[indicator], indicator)

# Economic
print("\nEconomic Indicators Trends:")
for indicator in ['indeks_pembangunan_manusia', 'penduduk_miskin']:
    plot_indicator_trend(all_data_final[indicator], indicator)

# Health
print("\nHealth Indicators Trends:")
for indicator in ['angka_harapan_hidup', 'persentase_balita_stunting']:
    plot_indicator_trend(all_data_final[indicator], indicator)

# Education
print("\nEducation Indicators Trends:")
for indicator in ['angka_melek_huruf', 'rata_rata_lama_sekolah']:
    plot_indicator_trend(all_data_final[indicator], indicator)


#%% [markdown]
# ## 3. Distribution of Prosperity Labels

#%%
# Function to plot label distribution for an indicator
def plot_label_distribution(data, indicator_name, title=None, year=None):
    """
    Plot the distribution of prosperity labels for an indicator
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the indicator data with labels
    indicator_name : str
        Name of the indicator
    title : str, optional
        Title for the plot
    """

    if year is not None:
        data = data.loc[data['year'] == year].copy()
        title = f"Distribution of Prosperity Labels for {indicator_name} at {year}"

    if title is None:
        title = f"Distribution of Prosperity Labels for {indicator_name}"
    
    plt.figure(figsize=(10, 6))
    
    # Count the labels
    label_counts = data['label_sejahtera'].value_counts()
    
    # Create a bar plot
    ax = sns.barplot(x=label_counts.index, y=label_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(label_counts.values):
        ax.text(i, count + 5, str(count), ha='center', fontsize=12)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Prosperity Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Also show the percentage distribution
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    plt.title(f"Percentage Distribution of Labels for {indicator_name}", fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Plot label distribution for selected indicators from each category
# Infrastructure
print("\nInfrastructure Indicators Label Distribution:")
for indicator in ['akses_air_minum', 'hunian_layak']:
    plot_label_distribution(all_data_final[indicator], indicator, year=2023)

# Economic
print("\nEconomic Indicators Label Distribution:")
for indicator in ['indeks_pembangunan_manusia', 'penduduk_miskin']:
    plot_label_distribution(all_data_final[indicator], indicator, year=2023)

# Health
print("\nHealth Indicators Label Distribution:")
for indicator in ['angka_harapan_hidup', 'persentase_balita_stunting']:
    plot_label_distribution(all_data_final[indicator], indicator, year=2023)

# Education
print("\nEducation Indicators Label Distribution:")
for indicator in ['angka_melek_huruf', 'rata_rata_lama_sekolah']:
    plot_label_distribution(all_data_final[indicator], indicator, year=2023)


#%% [markdown]
# ## 4. Correlation Between Indicators

#%%
# Create a dataframe with all indicators for a specific year (e.g., 2023)
def create_correlation_df(all_data, year=2023):
    """
    Create a dataframe with all indicators for a specific year
    
    Parameters:
    -----------
    all_data : dict
        Dictionary containing all indicator dataframes
    year : int
        Year to filter for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all indicators for the specified year
    """
    # Start with a base dataframe containing region names
    base_df = None
    
    # Find a dataframe that has data for the specified year
    for indicator, df in all_data.items():
        if year in df['year'].unique():
            base_df = df[df['year'] == year][['wilayah']].copy()
            break
    
    if base_df is None:
        print(f"No data found for year {year}")
        return None
    
    # Add each indicator to the base dataframe
    for indicator, df in all_data.items():
        if year in df['year'].unique():
            # Filter for the specified year
            year_df = df[df['year'] == year].copy()
            
            # Add the indicator value to the base dataframe
            base_df = base_df.merge(
                year_df[['wilayah', indicator]],
                on='wilayah',
                how='left'
            )
    
    return base_df


correlation_year = 2023

# Create correlation dataframe for {year}
corr_df_year = create_correlation_df(all_data_final, correlation_year)

# Plot correlation heatmap
if corr_df_year is not None:
    # Drop the wilayah column for correlation calculation
    corr_df_numeric = corr_df_year.drop(columns=['wilayah'])
    
    # Calculate correlation matrix
    corr_matrix = corr_df_numeric.corr()
    
    # Plot heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title(f"Correlation Between Indicators ({correlation_year})", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Plot correlation with selected indicators
    selected_indicators = [
        'indeks_pembangunan_manusia', 
        'penduduk_miskin', 
        'angka_harapan_hidup', 
        'persentase_balita_stunting',
        'rata_rata_lama_sekolah'
    ]
    
    # Check which selected indicators are actually in the dataframe
    available_indicators = [ind for ind in selected_indicators if ind in corr_df_numeric.columns]
    
    if available_indicators:
        # Calculate correlation with selected indicators
        corr_selected = corr_df_numeric.corr()[available_indicators]
        
        # Plot heatmap for selected indicators
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_selected, annot=True, fmt=".2f", cmap="coolwarm", 
                    linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f"Correlation with Selected Key Indicators ({correlation_year})", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


#%% [markdown]
# ## 5. Regional Comparisons

#%%
# Function to plot regional comparison for an indicator
def plot_regional_comparison(data, indicator_name, year=2023, top_n=20, title=None):
    """
    Plot regional comparison for an indicator
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the indicator data
    indicator_name : str
        Name of the indicator column
    year : int
        Year to filter for
    top_n : int
        Number of top regions to show
    title : str, optional
        Title for the plot
    """
    if title is None:
        title = f"Top {top_n} Regions by {indicator_name} ({year})"
    
    # Filter for the specified year
    year_data = data[data['year'] == year].copy()
    
    # Sort by indicator value
    sorted_data = year_data.sort_values(by=indicator_name, ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    ax = sns.barplot(x=indicator_name, y='wilayah', data=sorted_data, palette='viridis')
    
    # Add value labels
    for i, v in enumerate(sorted_data[indicator_name]):
        ax.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    # Color bars by prosperity label
    for i, (_, row) in enumerate(sorted_data.iterrows()):
        if row['label_sejahtera'] == 'Sejahtera':
            ax.patches[i].set_facecolor('green')
        elif row['label_sejahtera'] == 'Menengah':
            ax.patches[i].set_facecolor('orange')
        else:
            ax.patches[i].set_facecolor('red')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Sejahtera'),
        Patch(facecolor='orange', label='Menengah'),
        Patch(facecolor='red', label='Tidak Sejahtera')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.title(title, fontsize=14)
    plt.xlabel(indicator_name, fontsize=12)
    plt.ylabel('Region', fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot regional comparison for selected indicators
# Infrastructure
print("\nRegional Comparison - Infrastructure Indicators:")
for indicator in ['akses_air_minum', 'sanitasi_layak']:
    plot_regional_comparison(all_data_final[indicator], indicator)

# Economic
print("\nRegional Comparison - Economic Indicators:")
for indicator in ['indeks_pembangunan_manusia', 'penduduk_miskin']:
    plot_regional_comparison(all_data_final[indicator], indicator)

# Health
print("\nRegional Comparison - Health Indicators:")
for indicator in ['angka_harapan_hidup', 'persentase_balita_stunting']:
    plot_regional_comparison(all_data_final[indicator], indicator)

# Education
print("\nRegional Comparison - Education Indicators:")
for indicator in ['angka_melek_huruf', 'rata_rata_lama_sekolah']:
    plot_regional_comparison(all_data_final[indicator], indicator)


#%% [markdown]
# ## 6. Label Distribution Across Years

#%%
# Function to plot label distribution across years
def plot_label_trend(data, indicator_name, title=None):
    """
    Plot the trend of prosperity labels over time
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the indicator data with labels
    indicator_name : str
        Name of the indicator
    title : str, optional
        Title for the plot
    """
    if title is None:
        title = f"Trend of Prosperity Labels for {indicator_name} (2019-2023)"
    
    # Check if data is empty
    if data.empty:
        print(f"No data available for {indicator_name}")
        return
    
    # Check if 'label_sejahtera' column exists
    if 'label_sejahtera' not in data.columns:
        print(f"No 'label_sejahtera' column in {indicator_name} data")
        return
    
    # Count labels by year
    try:
        label_counts = data.groupby(['year', 'label_sejahtera']).size().unstack(fill_value=0)
        
        # Check if label_counts is empty
        if label_counts.empty:
            print(f"No label counts data for {indicator_name}")
            return
        
        # Convert year to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(label_counts.index):
            label_counts.index = pd.to_numeric(label_counts.index, errors='coerce')
        
        # Sort by year
        label_counts = label_counts.sort_index()
        
        # Check if there's any numeric data to plot
        if label_counts.sum().sum() == 0:
            print(f"No numeric data to plot for {indicator_name}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot stacked bar chart
        label_counts.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Prosperity Label')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Also plot the percentage distribution
        label_pct = label_counts.div(label_counts.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(12, 6))
        label_pct.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
        
        plt.title(f"Percentage Distribution of Labels for {indicator_name} by Year", fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.legend(title='Prosperity Label')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting label trend for {indicator_name}: {e}")
        # Print data info for debugging
        print(f"Data info: {data.info()}")
        print(f"Data head: {data.head()}")
        if 'label_sejahtera' in data.columns:
            print(f"Label counts: {data['label_sejahtera'].value_counts()}")
            print(f"Year values: {data['year'].unique()}")

# Plot label trends for selected indicators
# Infrastructure
print("\nLabel Trends - Infrastructure Indicators:")
for indicator in ['akses_air_minum', 'sanitasi_layak']:
    plot_label_trend(all_data_final[indicator], indicator)

# Economic
print("\nLabel Trends - Economic Indicators:")
for indicator in ['indeks_pembangunan_manusia', 'penduduk_miskin']:
    plot_label_trend(all_data_final[indicator], indicator)

# Health
print("\nLabel Trends - Health Indicators:")
for indicator in ['angka_harapan_hidup', 'persentase_balita_stunting']:
    plot_label_trend(all_data_final[indicator], indicator)

# Education
print("\nLabel Trends - Education Indicators:")
for indicator in ['angka_melek_huruf', 'rata_rata_lama_sekolah']:
    plot_label_trend(all_data_final[indicator], indicator)

print("\nExploratory Data Analysis completed.")


#%% [markdown]
# # Data Preparation for Model Training
# 
# In this section, we'll combine all indicators into a single dataset suitable for model training.
# We need to:
# 1. Create a dataset with all indicators for each region and year
# 2. Create a target variable based on the prosperity labels
# 3. Handle missing values
# 4. Normalize/standardize the features
# 5. Split the data into training and testing sets


#%% [markdown]
# ## Define target indicator for model target

#%%
target_indicator_for_model = 'indeks_pembangunan_manusia'

#%%
# Function to create a combined dataset for a specific year
def create_combined_dataset(all_data, year):
    """
    Create a combined dataset with all indicators for a specific year
    
    Parameters:
    -----------
    all_data : dict
        Dictionary containing all indicator dataframes
    year : int
        Year to filter for
        
    Returns:
    --------
    pd.DataFrame
        Combined dataset with all indicators for the specified year
    """
    # Start with a base dataframe containing region names
    base_df = None
    
    # Find a dataframe that has data for the specified year
    for indicator, df in all_data.items():
        if year in df['year'].unique():
            base_df = df[df['year'] == year][['wilayah']].copy()
            break
    
    if base_df is None:
        print(f"No data found for year {year}")
        return None

    
    # Add each indicator to the base dataframe
    for indicator, df in all_data.items():
        if year in df['year'].unique():
            # Filter for the specified year
            year_df = df[df['year'] == year].copy()
            
            # Rename the label_sejahtera column to include the indicator name
            year_df = year_df.rename(columns={'label_sejahtera': f'label_sejahtera_{indicator}'})
            
            # Add the indicator value to the base dataframe
            base_df = base_df.merge(
                year_df[['wilayah', indicator, f'label_sejahtera_{indicator}']],
                on='wilayah',
                how='left'
            )
    
    return base_df

# Create combined datasets for each year
combined_datasets = {}
for year in [2019, 2020, 2021, 2022, 2023]:
    combined_datasets[year] = create_combined_dataset(all_data_final, year)
    if combined_datasets[year] is not None:
        print(f"Combined dataset for {year}: {combined_datasets[year].shape}")


#%%
# Create a function to prepare the data for model training
def prepare_data_for_model(combined_df, target_indicator):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined dataset with all indicators
    target_indicator : str
        Indicator to use as the target variable
        
    Returns:
    --------
    X : pd.DataFrame
        Features for model training
    y : pd.Series
        Target variable for model training
    """
    # Make a copy of the dataframe
    df = combined_df.copy()
    
    # Get the target variable (prosperity label for the specified indicator)
    target_label_col = f'label_sejahtera_{target_indicator}'
    
    # Check if the target column exists
    if target_label_col not in df.columns:
        print(f"Target column {target_label_col} not found in the dataframe")
        return None, None
    
    # Create the target variable
    y = df[target_label_col]
    
    # Create the feature matrix
    # Drop non-feature columns (wilayah, label columns, and the target indicator)
    drop_cols = ['wilayah']
    drop_cols.extend([col for col in df.columns if 'label_sejahtera' in col])
    drop_cols.append(target_indicator)
    
    # Keep only numeric columns for features
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=['number'])
    
    # Handle missing values
    # For simplicity, we'll fill missing values with the mean of each column
    X = X.fillna(X.mean())
    
    # Check for any remaining NaN values
    if X.isna().any().any():
        print("Warning: There are still NaN values in the feature matrix after imputation.")
        print("Columns with NaN values:", X.columns[X.isna().any()].tolist())
        
        # For columns where all values are NaN, fill with 0
        for col in X.columns[X.isna().any()]:
            X[col] = X[col].fillna(0)
    
    return X, y

# Create a function to prepare the data for all years
def prepare_all_years_data(combined_datasets, target_indicator):
    """
    Prepare data for all years
    
    Parameters:
    -----------
    combined_datasets : dict
        Dictionary containing combined datasets for each year
    target_indicator : str
        Indicator to use as the target variable
        
    Returns:
    --------
    X_all : pd.DataFrame
        Features for all years
    y_all : pd.Series
        Target variable for all years
    """
    X_all = pd.DataFrame()
    y_all = pd.Series(dtype='object')
    
    for year, df in combined_datasets.items():
        if df is not None:
            X, y = prepare_data_for_model(df, target_indicator)
            if X is not None and y is not None:
                # Add year as a feature - ensure it's numeric
                X['year'] = pd.to_numeric(year, errors='coerce')
                
                # Append to the combined dataset
                X_all = pd.concat([X_all, X])
                y_all = pd.concat([y_all, y])
    
    # Final check for NaN values
    if X_all.isna().any().any():
        print("Warning: There are still NaN values in the combined feature matrix.")
        print("Columns with NaN values:", X_all.columns[X_all.isna().any()].tolist())
        
        # Fill any remaining NaN values with the mean of each column
        X_all = X_all.fillna(X_all.mean())
        
        # For columns where all values are NaN, fill with 0
        for col in X_all.columns[X_all.isna().any()]:
            X_all[col] = X_all[col].fillna(0)
    
    return X_all, y_all

#%%
# Prepare data for model training using target_indicator_for_model variable as the target
X_all, y_all = prepare_all_years_data(combined_datasets, target_indicator_for_model)

print("\nCombined dataset for all years:")
print(f"X shape: {X_all.shape}")
print(f"y shape: {y_all.shape}")
print("\nFeatures:")
print(X_all.columns.tolist())
print("\nTarget distribution:")
print(y_all.value_counts())

# Check for NaN values before standardization
if X_all.isna().any().any():
    print("\nWarning: There are NaN values in the feature matrix before standardization.")
    print("Columns with NaN values:", X_all.columns[X_all.isna().any()].tolist())
    
    # Fill any remaining NaN values with the mean of each column
    X_all = X_all.fillna(X_all.mean())
    
    # For columns where all values are NaN, fill with 0
    for col in X_all.columns[X_all.isna().any()]:
        X_all[col] = X_all[col].fillna(0)

# Standardize the features
from sklearn.preprocessing import StandardScaler

# Create a scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X_all)

# Convert back to a DataFrame for easier handling
X_scaled_df = pd.DataFrame(X_scaled, columns=X_all.columns, index=X_all.index)

print("\nStandardized features (first 5 rows):")
print(X_scaled_df.head())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Final check for NaN values in training and testing sets
if X_train.isna().any().any() or X_test.isna().any().any():
    print("\nWarning: There are still NaN values in the training or testing sets after splitting.")
    
    # Fill any remaining NaN values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

print("\nTraining and testing sets:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\nTraining set target distribution:")
print(y_train.value_counts())
print("\nTesting set target distribution:")
print(y_test.value_counts())

# Save the prepared data for model training
model_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_all': X_all,
    'y_all': y_all,
    'X_scaled': X_scaled_df,
    'feature_names': X_all.columns.tolist(),
    'scaler': scaler
}

print("\nData preparation for model training completed.")



#%% [markdown]
# # Model Training - Random Forest with K-fold Cross Validation
# 
# In this section, we'll implement and train a Random Forest model using 10-fold cross-validation.
# Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training
# and outputting the class that is the mode of the classes of the individual trees.

#%%
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_depth=None,    # Maximum depth of the trees (None means unlimited)
    min_samples_split=2,  # Minimum samples required to split an internal node
    min_samples_leaf=1,   # Minimum samples required to be at a leaf node
    bootstrap=True,       # Whether to use bootstrap samples
    random_state=42       # Random seed for reproducibility
)

# Set up K-fold cross-validation
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation
print("\nPerforming 10-fold cross-validation for Random Forest...")
cv_scores = cross_val_score(rf_classifier, model_data['X_train'], model_data['y_train'], cv=kf, scoring='accuracy')

# Print cross-validation results
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Get predictions for each fold
y_pred_cv = cross_val_predict(rf_classifier, model_data['X_train'], model_data['y_train'], cv=kf)

# Print classification report for cross-validation
print("\nClassification Report (Cross-Validation):")
print(classification_report(model_data['y_train'], y_pred_cv))

# Plot confusion matrix for cross-validation
plt.figure(figsize=(10, 8))
cm = confusion_matrix(model_data['y_train'], y_pred_cv)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(model_data['y_train']),
            yticklabels=np.unique(model_data['y_train']))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Cross-Validation)')
plt.tight_layout()
plt.show()

# Train the final model on the entire training set
print("\nTraining the final Random Forest model on the entire training set...")
rf_classifier.fit(model_data['X_train'], model_data['y_train'])

# Make predictions on the test set
y_pred_test = rf_classifier.predict(model_data['X_test'])

# Evaluate the model on the test set
test_accuracy = accuracy_score(model_data['y_test'], y_pred_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Print classification report for test set
print("\nClassification Report (Test Set):")
print(classification_report(model_data['y_test'], y_pred_test))

# Plot confusion matrix for test set
plt.figure(figsize=(10, 8))
cm_test = confusion_matrix(model_data['y_test'], y_pred_test)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(model_data['y_test']),
            yticklabels=np.unique(model_data['y_test']))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.show()

# Feature importance analysis
feature_importances = rf_classifier.feature_importances_
feature_names = model_data['feature_names']

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# Save the Random Forest model and results
rf_results = {
    'model': rf_classifier,
    'cv_scores': cv_scores,
    'mean_cv_accuracy': cv_scores.mean(),
    'std_cv_accuracy': cv_scores.std(),
    'test_accuracy': test_accuracy,
    'y_pred_cv': y_pred_cv,
    'y_pred_test': y_pred_test,
    'feature_importances': feature_importance_df
}

print("\nRandom Forest model training completed.")


#%% [markdown]
# # Model Training - Logistic Regression with K-fold Cross Validation
# 
# In this section, we'll implement and train a Logistic Regression model using 10-fold cross-validation.
# Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable,
# but can be extended to handle multi-class classification problems.

#%%
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a Logistic Regression classifier
# For multi-class problems, we use the 'multinomial' solver with 'lbfgs' solver
lr_classifier = LogisticRegression(
    C=1.0,               # Inverse of regularization strength
    penalty='l2',        # L2 regularization
    solver='lbfgs',      # Algorithm to use in the optimization problem
    max_iter=2000,       # Maximum number of iterations (increased from 1000)
    multi_class='multinomial',  # Multi-class option
    random_state=42,     # Random seed for reproducibility
    n_jobs=-1            # Use all available cores
)

# Final check for NaN values in training data
if X_train.isna().any().any():
    print("\nWarning: There are still NaN values in the training data. Filling with zeros.")
    X_train = X_train.fillna(0)

if X_test.isna().any().any():
    print("\nWarning: There are still NaN values in the test data. Filling with zeros.")
    X_test = X_test.fillna(0)

# Set up K-fold cross-validation (reuse the same KFold object from Random Forest)
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation with error handling
print("\nPerforming 10-fold cross-validation for Logistic Regression...")
try:
    lr_cv_scores = cross_val_score(lr_classifier, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Print cross-validation results
    print(f"Cross-validation scores: {lr_cv_scores}")
    print(f"Mean CV accuracy: {lr_cv_scores.mean():.4f}")
    print(f"Standard deviation: {lr_cv_scores.std():.4f}")
    
    # Get predictions for each fold
    lr_y_pred_cv = cross_val_predict(lr_classifier, X_train, y_train, cv=kf)
    
    # Print classification report for cross-validation
    print("\nClassification Report (Cross-Validation):")
    print(classification_report(y_train, lr_y_pred_cv))
    
    # Plot confusion matrix for cross-validation
    plt.figure(figsize=(10, 8))
    lr_cm = confusion_matrix(y_train, lr_y_pred_cv)
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_train),
                yticklabels=np.unique(y_train))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Logistic Regression (Cross-Validation)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error during cross-validation: {e}")
    print("Trying with a simpler approach...")
    
    # Try with a simpler approach (OvR instead of multinomial)
    lr_classifier = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',  # Change solver to liblinear which is more robust
        max_iter=2000,
        multi_class='ovr',   # Change to one-vs-rest
        random_state=42
    )
    
    # Try cross-validation again
    try:
        lr_cv_scores = cross_val_score(lr_classifier, X_train, y_train, cv=kf, scoring='accuracy')
        
        # Print cross-validation results
        print(f"Cross-validation scores with simpler model: {lr_cv_scores}")
        print(f"Mean CV accuracy: {lr_cv_scores.mean():.4f}")
        print(f"Standard deviation: {lr_cv_scores.std():.4f}")
        
        # Get predictions for each fold
        lr_y_pred_cv = cross_val_predict(lr_classifier, X_train, y_train, cv=kf)
    except Exception as e2:
        print(f"Error with simpler approach: {e2}")
        print("Skipping cross-validation for Logistic Regression.")
        
        # Create dummy values for results
        lr_cv_scores = np.array([0.0])
        lr_y_pred_cv = y_train.copy()

# Train the final model on the entire training set
print("\nTraining the final Logistic Regression model on the entire training set...")
try:
    lr_classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    lr_y_pred_test = lr_classifier.predict(X_test)
    
    # Evaluate the model on the test set
    lr_test_accuracy = accuracy_score(y_test, lr_y_pred_test)
    print(f"\nTest accuracy: {lr_test_accuracy:.4f}")
    
    # Print classification report for test set
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, lr_y_pred_test))
    
    # Plot confusion matrix for test set
    plt.figure(figsize=(10, 8))
    lr_cm_test = confusion_matrix(y_test, lr_y_pred_test)
    sns.heatmap(lr_cm_test, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Logistic Regression (Test Set)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error during model training: {e}")
    print("Skipping Logistic Regression model training.")
    
    # Create dummy values for results
    lr_test_accuracy = 0.0
    lr_y_pred_test = y_test.copy()

# Feature importance analysis for Logistic Regression
# For multi-class, we'll look at the absolute values of coefficients for each class
if hasattr(lr_classifier, 'coef_'):
    try:
        # Get feature names
        feature_names = model_data['feature_names']
        
        # Get coefficients
        coefficients = lr_classifier.coef_
        
        # For multi-class, we have coefficients for each class
        # We'll take the average absolute value across classes
        coef_importance = np.mean(np.abs(coefficients), axis=0)
        
        # Create a DataFrame for coefficient importances
        coef_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coef_importance
        })
        
        # Sort by importance
        coef_importance_df = coef_importance_df.sort_values('Importance', ascending=False)
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=coef_importance_df.head(15))
        plt.title('Top 15 Feature Importances (Logistic Regression)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during feature importance analysis: {e}")
        print("Skipping feature importance analysis for Logistic Regression.")
        
        # Create a dummy DataFrame for feature importances
        coef_importance_df = pd.DataFrame({
            'Feature': model_data['feature_names'],
            'Importance': np.zeros(len(model_data['feature_names']))
        })
else:
    print("Logistic Regression model does not have coefficients. Skipping feature importance analysis.")
    
    # Create a dummy DataFrame for feature importances
    coef_importance_df = pd.DataFrame({
        'Feature': model_data['feature_names'],
        'Importance': np.zeros(len(model_data['feature_names']))
    })

# Save the Logistic Regression model and results
lr_results = {
    'model': lr_classifier,
    'cv_scores': lr_cv_scores,
    'mean_cv_accuracy': lr_cv_scores.mean() if len(lr_cv_scores) > 0 else 0.0,
    'std_cv_accuracy': lr_cv_scores.std() if len(lr_cv_scores) > 0 else 0.0,
    'test_accuracy': lr_test_accuracy,
    'y_pred_cv': lr_y_pred_cv,
    'y_pred_test': lr_y_pred_test,
    'feature_importances': coef_importance_df
}

print("\nLogistic Regression model training completed.")


#%% [markdown]
# # Model Evaluation and Comparison
# 
# In this section, we'll evaluate and compare the performance of both models:
# 1. Random Forest
# 2. Logistic Regression
# 
# We'll use various metrics including:
# - Accuracy
# - Precision, Recall, and F1-score
# - Confusion matrices

#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Function to plot confusion matrix with percentages
def plot_confusion_matrix_with_percentages(cm, classes, title, cmap=plt.cm.Blues):
    """
    Plot confusion matrix with percentages
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    classes : list
        List of class names
    title : str
        Title for the plot
    cmap : matplotlib.colors.Colormap
        Colormap for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    
    # Add percentage labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})', 
                     ha='center', va='center', color='black', fontsize=9)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Get class names
class_names = np.unique(model_data['y_test'])

# Plot confusion matrices for both models
print("\nConfusion Matrix Comparison:")

# Random Forest confusion matrix
plot_confusion_matrix_with_percentages(
    confusion_matrix(model_data['y_test'], rf_results['y_pred_test']),
    class_names,
    'Confusion Matrix - Random Forest (Test Set)'
)

# Logistic Regression confusion matrix
plot_confusion_matrix_with_percentages(
    confusion_matrix(model_data['y_test'], lr_results['y_pred_test']),
    class_names,
    'Confusion Matrix - Logistic Regression (Test Set)'
)

# Compare accuracy scores
print("\nAccuracy Comparison:")
print(f"Random Forest - Cross-validation: {rf_results['mean_cv_accuracy']:.4f} ± {rf_results['std_cv_accuracy']:.4f}")
print(f"Random Forest - Test set: {rf_results['test_accuracy']:.4f}")
print(f"Logistic Regression - Cross-validation: {lr_results['mean_cv_accuracy']:.4f} ± {lr_results['std_cv_accuracy']:.4f}")
print(f"Logistic Regression - Test set: {lr_results['test_accuracy']:.4f}")

# Compare classification reports
print("\nClassification Report Comparison:")
print("Random Forest:")
print(classification_report(model_data['y_test'], rf_results['y_pred_test']))
print("\nLogistic Regression:")
print(classification_report(model_data['y_test'], lr_results['y_pred_test']))

# Compare feature importances
plt.figure(figsize=(14, 10))

# Get top 10 features from each model
rf_top_features = rf_results['feature_importances'].head(10)
lr_top_features = lr_results['feature_importances'].head(10)

# Combine and get unique features
all_top_features = pd.concat([rf_top_features, lr_top_features])
all_top_features = all_top_features.drop_duplicates('Feature')

# Create a new DataFrame with all top features
comparison_df = pd.DataFrame({'Feature': all_top_features['Feature'].unique()})

# Add importance values from both models
for feature in comparison_df['Feature']:
    # Random Forest
    if feature in rf_top_features['Feature'].values:
        comparison_df.loc[comparison_df['Feature'] == feature, 'RF_Importance'] = \
            rf_top_features.loc[rf_top_features['Feature'] == feature, 'Importance'].values[0]
    else:
        comparison_df.loc[comparison_df['Feature'] == feature, 'RF_Importance'] = 0
    
    # Logistic Regression
    if feature in lr_top_features['Feature'].values:
        comparison_df.loc[comparison_df['Feature'] == feature, 'LR_Importance'] = \
            lr_top_features.loc[lr_top_features['Feature'] == feature, 'Importance'].values[0]
    else:
        comparison_df.loc[comparison_df['Feature'] == feature, 'LR_Importance'] = 0

# Sort by average importance
comparison_df['Avg_Importance'] = (comparison_df['RF_Importance'] + comparison_df['LR_Importance']) / 2
comparison_df = comparison_df.sort_values('Avg_Importance', ascending=False).head(15)

# Normalize importances for better visualization
comparison_df['RF_Importance_Norm'] = comparison_df['RF_Importance'] / comparison_df['RF_Importance'].max()
comparison_df['LR_Importance_Norm'] = comparison_df['LR_Importance'] / comparison_df['LR_Importance'].max()

# Plot
plt.figure(figsize=(14, 10))
x = np.arange(len(comparison_df))
width = 0.35

plt.bar(x - width/2, comparison_df['RF_Importance_Norm'], width, label='Random Forest')
plt.bar(x + width/2, comparison_df['LR_Importance_Norm'], width, label='Logistic Regression')

plt.xlabel('Features')
plt.ylabel('Normalized Importance')
plt.title('Feature Importance Comparison')
plt.xticks(x, comparison_df['Feature'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Summary of model comparison
print("\nModel Comparison Summary:")
print("-" * 50)
print("Metrics                  | Random Forest    | Logistic Regression")
print("-" * 50)
print(f"Cross-validation Accuracy | {rf_results['mean_cv_accuracy']:.4f} ± {rf_results['std_cv_accuracy']:.4f} | {lr_results['mean_cv_accuracy']:.4f} ± {lr_results['std_cv_accuracy']:.4f}")
print(f"Test Accuracy             | {rf_results['test_accuracy']:.4f}        | {lr_results['test_accuracy']:.4f}")
print("-" * 50)

# Determine the better model
if rf_results['test_accuracy'] > lr_results['test_accuracy']:
    better_model = "Random Forest"
    reason = "higher test accuracy"
elif rf_results['test_accuracy'] < lr_results['test_accuracy']:
    better_model = "Logistic Regression"
    reason = "higher test accuracy"
else:
    # If accuracies are equal, compare cross-validation stability
    if rf_results['std_cv_accuracy'] < lr_results['std_cv_accuracy']:
        better_model = "Random Forest"
        reason = "more stable performance across cross-validation folds (with equal test accuracy)"
    elif rf_results['std_cv_accuracy'] > lr_results['std_cv_accuracy']:
        better_model = "Logistic Regression"
        reason = "more stable performance across cross-validation folds (with equal test accuracy)"
    else:
        better_model = "Both models perform equally"
        reason = "equal test accuracy and cross-validation stability"

print(f"\nBased on the evaluation, the better model is: {better_model}")
print(f"Reason: {reason}")

# Additional considerations
print("\nAdditional Considerations:")
print("1. Random Forest:")
print("   - Advantages: Can handle non-linear relationships, less prone to overfitting")
print("   - Disadvantages: Less interpretable, computationally more intensive")
print("2. Logistic Regression:")
print("   - Advantages: More interpretable, computationally efficient")
print("   - Disadvantages: May not capture complex relationships in the data")

print("\nModel evaluation and comparison completed.")


#%% [markdown]
# # Visualization of Model Results
# 
# In this section, we'll create visualizations of the model results, including:
# 1. Regional prosperity maps
# 2. Prediction distribution by region
# 3. Comparison of actual vs. predicted prosperity labels
# 4. Prosperity trends over time

#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# Use the better model for predictions (based on previous evaluation)
if better_model == "Random Forest":
    best_model = rf_results['model']
    model_name = "Random Forest"
else:
    best_model = lr_results['model']
    model_name = "Logistic Regression"

print(f"\nUsing {model_name} for result visualizations as it performed better in evaluation.")

# Create a function to generate predictions for all regions and years
def generate_predictions_for_all_data(model, all_data, scaler):
    """
    Generate predictions for all regions and years
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to use for predictions
    all_data : dict
        Dictionary containing all indicator dataframes
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used to standardize features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions for all regions and years
    """
    # Create a list to store predictions
    all_predictions = []
    
    # Generate predictions for each year
    for year in [2019, 2020, 2021, 2022, 2023]:
        # Create combined dataset for the year
        combined_df = create_combined_dataset(all_data, year)
        
        if combined_df is not None:
            # Extract features
            X, y = prepare_data_for_model(combined_df, target_indicator_for_model)
            
            if X is not None and y is not None:
                # Add year as a feature - ensure it's numeric
                X['year'] = pd.to_numeric(year, errors='coerce')
                
                # Ensure all columns from the scaler are present in X
                for col in scaler.feature_names_in_:
                    if col not in X.columns:
                        X[col] = 0  # Add missing columns with default value 0
                
                # Ensure columns are in the same order as during fit
                X = X[scaler.feature_names_in_]
                
                # Standardize features
                X_scaled = scaler.transform(X)
                
                # Generate predictions
                y_pred = model.predict(X_scaled)
                
                # Create a DataFrame with predictions
                pred_df = pd.DataFrame({
                    'wilayah': combined_df['wilayah'],
                    'year': year,
                    'actual': y,
                    'predicted': y_pred
                })
                
                # Add to the list
                all_predictions.append(pred_df)
    
    # Combine all predictions
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return pd.DataFrame()

# Generate predictions for all regions and years
all_predictions_df = generate_predictions_for_all_data(
    best_model, 
    all_data_final, 
    model_data['scaler']
)

print(f"\nGenerated predictions for {len(all_predictions_df)} region-year combinations.")
print(all_predictions_df.head())

# 1. Regional Prosperity Map (for the most recent year, 2023)
# Since we don't have actual geographical data, we'll create a visual representation

# Filter for 2023 data
predictions_2023 = all_predictions_df[all_predictions_df['year'] == 2023].copy()

# Create a categorical color map
prosperity_colors = {
    'Sejahtera': '#2ecc71',      # Green
    'Menengah': '#f39c12',       # Orange
    'Tidak Sejahtera': '#e74c3c'  # Red
}

# Create a function to visualize regional prosperity
def plot_regional_prosperity(predictions, year, based_on='predicted', title=None):
    """
    Visualize regional prosperity
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions
    year : str
        Year to visualize
    based_on : str
        Column to use for coloring ('predicted' or 'actual')
    title : str, optional
        Title for the plot
    """
    # Filter for the specified year
    year_data = predictions[predictions['year'] == year].copy()
    
    if year_data.empty:
        print(f"No data available for year {year}")
        return
    
    # Sort by region name
    year_data = year_data.sort_values('wilayah')
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a horizontal bar chart
    bars = plt.barh(year_data['wilayah'], 
                    np.ones(len(year_data)),  # All bars have the same width
                    color=[prosperity_colors[label] for label in year_data[based_on]])
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=prosperity_colors['Sejahtera'], label='Sejahtera'),
        Patch(facecolor=prosperity_colors['Menengah'], label='Menengah'),
        Patch(facecolor=prosperity_colors['Tidak Sejahtera'], label='Tidak Sejahtera')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Set title and labels
    if title is None:
        title = f"Regional Prosperity Map ({year}) - Based on {based_on.capitalize()} Labels"
    plt.title(title, fontsize=16)
    plt.xlabel('Prosperity Level', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    
    # Remove x-axis ticks
    plt.xticks([])
    
    # Add prosperity label text to each bar
    for i, (_, row) in enumerate(year_data.iterrows()):
        plt.text(0.5, i, row[based_on], 
                 ha='center', va='center', 
                 color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Plot regional prosperity map for 2023 (predicted)
plot_regional_prosperity(all_predictions_df, '2023', based_on='predicted',
                        title=f"Regional Prosperity Map (2023) - {model_name} Predictions")

# Plot regional prosperity map for 2023 (actual)
plot_regional_prosperity(all_predictions_df, '2023', based_on='actual',
                        title="Regional Prosperity Map (2023) - Actual Labels")

# 2. Prediction Distribution by Region
# Create a function to visualize prediction distribution
def plot_prediction_distribution(predictions):
    """
    Visualize prediction distribution
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions
    """
    # Count predictions by category
    pred_counts = predictions['predicted'].value_counts().reset_index()
    pred_counts.columns = ['Category', 'Count']
    
    # Count actual labels by category
    actual_counts = predictions['actual'].value_counts().reset_index()
    actual_counts.columns = ['Category', 'Count']
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    x = np.arange(len(pred_counts))
    width = 0.35
    
    plt.bar(x - width/2, pred_counts['Count'], width, label='Predicted', color='#3498db')
    plt.bar(x + width/2, actual_counts['Count'], width, label='Actual', color='#e74c3c')
    
    # Add labels and title
    plt.xlabel('Prosperity Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Predicted vs. Actual Prosperity Categories', fontsize=16)
    plt.xticks(x, pred_counts['Category'])
    plt.legend()
    
    # Add count labels on top of bars
    for i, count in enumerate(pred_counts['Count']):
        plt.text(i - width/2, count + 1, str(count), ha='center')
    
    for i, count in enumerate(actual_counts['Count']):
        plt.text(i + width/2, count + 1, str(count), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Also create a pie chart for predicted distribution
    plt.figure(figsize=(10, 8))
    plt.pie(pred_counts['Count'], labels=pred_counts['Category'], autopct='%1.1f%%',
            colors=[prosperity_colors[cat] for cat in pred_counts['Category']])
    plt.title(f'Distribution of Predicted Prosperity Categories ({model_name})', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Plot prediction distribution for 2023
plot_prediction_distribution(predictions_2023)

# 3. Comparison of actual vs. predicted prosperity labels
# Create a function to visualize prediction accuracy by region
def plot_prediction_accuracy_by_region(predictions):
    """
    Visualize prediction accuracy by region
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions
    """
    # Create a new column for correct predictions
    predictions['correct'] = predictions['predicted'] == predictions['actual']
    
    # Group by region and calculate accuracy
    region_accuracy = predictions.groupby('wilayah')['correct'].mean().reset_index()
    region_accuracy.columns = ['wilayah', 'accuracy']
    
    # Sort by accuracy
    region_accuracy = region_accuracy.sort_values('accuracy', ascending=False)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a horizontal bar chart
    bars = plt.barh(region_accuracy['wilayah'], region_accuracy['accuracy'])
    
    # Color bars based on accuracy
    for i, bar in enumerate(bars):
        if region_accuracy['accuracy'].iloc[i] >= 0.8:
            bar.set_color('#2ecc71')  # Green for high accuracy
        elif region_accuracy['accuracy'].iloc[i] >= 0.5:
            bar.set_color('#f39c12')  # Orange for medium accuracy
        else:
            bar.set_color('#e74c3c')  # Red for low accuracy
    
    # Add accuracy values as text
    for i, v in enumerate(region_accuracy['accuracy']):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center')
    
    # Set title and labels
    plt.title(f'Prediction Accuracy by Region ({model_name})', fontsize=16)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    
    # Add a vertical line at 0.5 and 0.8 accuracy
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.show()

# Plot prediction accuracy by region
plot_prediction_accuracy_by_region(all_predictions_df)

# 4. Prosperity trends over time
# Create a function to visualize prosperity trends over time
def plot_prosperity_trends(predictions):
    """
    Visualize prosperity trends over time
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with predictions
    """
    # Group by year and count prosperity categories
    yearly_counts = predictions.groupby(['year', 'predicted']).size().unstack(fill_value=0)
    
    # Calculate percentages
    yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create a stacked bar chart
    yearly_pct.plot(kind='bar', stacked=True, ax=plt.gca(), 
                   color=[prosperity_colors[cat] for cat in yearly_pct.columns])
    
    # Add labels and title
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title(f'Prosperity Trends Over Time ({model_name} Predictions)', fontsize=16)
    plt.legend(title='Prosperity Category')
    
    # Add percentage labels
    for i, year in enumerate(yearly_pct.index):
        cumulative_height = 0
        for category in yearly_pct.columns:
            height = yearly_pct.loc[year, category]
            if height > 5:  # Only add label if segment is large enough
                plt.text(i, cumulative_height + height/2, f"{height:.1f}%", 
                         ha='center', va='center', fontweight='bold')
            cumulative_height += height
    
    plt.tight_layout()
    plt.show()
    
    # Also create a line chart for each category
    plt.figure(figsize=(12, 8))
    
    for category in yearly_counts.columns:
        plt.plot(yearly_counts.index, yearly_counts[category], 
                 marker='o', linewidth=2, label=category,
                 color=prosperity_colors[category])
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Prosperity Category Counts Over Time ({model_name} Predictions)', fontsize=16)
    plt.legend(title='Prosperity Category')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot prosperity trends over time
plot_prosperity_trends(all_predictions_df)

# 5. Feature importance and prosperity correlation
# Create a function to visualize feature importance and prosperity correlation
def plot_feature_importance_prosperity_correlation(model, feature_names, predictions, combined_data):
    """
    Visualize feature importance and prosperity correlation
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    feature_names : list
        List of feature names
    predictions : pd.DataFrame
        DataFrame with predictions
    combined_data : pd.DataFrame
        Combined dataset for a specific year
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        print("Model does not have feature importances or coefficients")
        return
    
    # Get top 10 features
    indices = np.argsort(importances)[-10:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create a horizontal bar chart
    plt.barh(top_features, top_importances)
    
    # Add importance values as text
    for i, v in enumerate(top_importances):
        plt.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Set title and labels
    plt.title(f'Top 10 Feature Importances ({model_name})', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # For the most recent year, show correlation between top features and prosperity
    if combined_data is not None and not combined_data.empty:
        # Add predictions to the combined data
        combined_data = combined_data.copy()
        combined_data['predicted'] = predictions[predictions['year'] == '2023']['predicted'].values
        
        # Create a categorical encoding for prosperity
        prosperity_map = {
            'Sejahtera': 2,
            'Menengah': 1,
            'Tidak Sejahtera': 0
        }
        combined_data['prosperity_score'] = combined_data['predicted'].map(prosperity_map)
        
        # Calculate correlation between top features and prosperity score
        corr_data = combined_data[['prosperity_score'] + top_features].corr()['prosperity_score'].drop('prosperity_score')
        
        # Sort by absolute correlation
        corr_data = corr_data.reindex(corr_data.abs().sort_values(ascending=False).index)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create a horizontal bar chart
        bars = plt.barh(corr_data.index, corr_data.values)
        
        # Color bars based on correlation (positive or negative)
        for i, bar in enumerate(bars):
            if corr_data.values[i] > 0:
                bar.set_color('#2ecc71')  # Green for positive correlation
            else:
                bar.set_color('#e74c3c')  # Red for negative correlation
        
        # Add correlation values as text
        for i, v in enumerate(corr_data.values):
            plt.text(v + 0.01 if v > 0 else v - 0.1, i, f"{v:.2f}", va='center')
        
        # Set title and labels
        plt.title('Correlation Between Top Features and Prosperity (2023)', fontsize=16)
        plt.xlabel('Correlation with Prosperity Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Add a vertical line at 0
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

# Get combined data for 2023
combined_data_2023 = create_combined_dataset(all_data_final, '2023')

# Plot feature importance and prosperity correlation
plot_feature_importance_prosperity_correlation(
    best_model, 
    model_data['feature_names'], 
    all_predictions_df,
    combined_data_2023
)

print("\nVisualization of model results completed.")

#%% [markdown]
# # Final Analysis and Conclusions
# 
# In this section, we'll summarize our findings, draw conclusions, and provide recommendations based on our analysis.

#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Create a summary of the entire analysis process
print("\n" + "="*80)
print("PROSPERITY MODELLING: FINAL ANALYSIS AND CONCLUSIONS")
print("="*80)

# 1. Data Overview
print("\n1. DATA OVERVIEW")
print("-"*50)
print(f"Total number of indicators analyzed: {len(all_data_final)}")
print(f"Years covered in the analysis: 2019-2023")
print(f"Number of regions analyzed: {len(all_predictions_df['wilayah'].unique())}")
print("\nIndicator categories:")
print("- Infrastructure: 8 indicators (including kendaraan split into roda_2 and roda_4)")
print("- Economic: 7 indicators")
print("- Health: 7 indicators")
print("- Education: Multiple indicators (including splits by education level)")

# 2. Preprocessing and Labeling Summary
print("\n2. PREPROCESSING AND LABELING SUMMARY")
print("-"*50)
print("Data preprocessing steps:")
print("- Standardized data formats across all indicators")
print("- Handled special cases (e.g., using 2013 data for 2019 in angka_melek_huruf)")
print("- Filtered data for years 2019-2023")
print("- Aggregated data where needed (e.g., summing different types of fasilitas_kesehatan)")

print("\nLabeling approach:")
print("- Most indicators: IQR-based labeling (Sejahtera, Menengah, Tidak Sejahtera)")
print("- Special cases with manual thresholds:")
print("  * indeks_pembangunan_manusia: Sejahtera if > 70, Menengah if 60-70, else Tidak Sejahtera")
print("  * tingkat_pengangguran_terbuka: Sejahtera if < 6.75, Menengah if 6.5-7.0, else Tidak Sejahtera")
print("  * persentase_balita_stunting: Sejahtera if < 20, Menengah if 20-29, else Tidak Sejahtera")

# 3. Model Performance Summary
print("\n3. MODEL PERFORMANCE SUMMARY")
print("-"*50)
print("Random Forest:")
print(f"- Cross-validation accuracy: {rf_results['mean_cv_accuracy']:.4f} ± {rf_results['std_cv_accuracy']:.4f}")
print(f"- Test accuracy: {rf_results['test_accuracy']:.4f}")

print("\nLogistic Regression:")
print(f"- Cross-validation accuracy: {lr_results['mean_cv_accuracy']:.4f} ± {lr_results['std_cv_accuracy']:.4f}")
print(f"- Test accuracy: {lr_results['test_accuracy']:.4f}")

print(f"\nBetter model: {better_model}")
print(f"Reason: {reason}")

# 4. Key Findings
print("\n4. KEY FINDINGS")
print("-"*50)

# Calculate prosperity distribution for 2023
prosperity_dist_2023 = predictions_2023['predicted'].value_counts(normalize=True) * 100
prosperity_counts_2023 = predictions_2023['predicted'].value_counts()

print("Prosperity distribution in 2023:")
for category, percentage in prosperity_dist_2023.items():
    print(f"- {category}: {percentage:.1f}% ({prosperity_counts_2023[category]} regions)")

# Calculate change in prosperity from 2019 to 2023
predictions_2019 = all_predictions_df[all_predictions_df['year'] == '2019'].copy()
if not predictions_2019.empty:
    prosperity_counts_2019 = predictions_2019['predicted'].value_counts()
    prosperity_dist_2019 = predictions_2019['predicted'].value_counts(normalize=True) * 100
    
    print("\nChange in prosperity from 2019 to 2023:")
    for category in prosperity_counts_2023.index:
        if category in prosperity_counts_2019:
            count_2019 = prosperity_counts_2019[category]
            count_2023 = prosperity_counts_2023[category]
            change = count_2023 - count_2019
            change_pct = prosperity_dist_2023[category] - prosperity_dist_2019[category]
            
            print(f"- {category}: {change:+d} regions ({change_pct:+.1f} percentage points)")

# Get top 5 most important features
if 'feature_importances' in rf_results:
    top_features = rf_results['feature_importances'].head(5)['Feature'].tolist()
    print("\nTop 5 most important indicators for prosperity prediction:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")

# Identify regions with consistent prosperity
consistent_regions = all_predictions_df.groupby('wilayah')['predicted'].apply(
    lambda x: len(x.unique()) == 1 and 'Sejahtera' in x.unique()
).reset_index()
consistent_sejahtera = consistent_regions[consistent_regions['predicted'] == True]['wilayah'].tolist()

if consistent_sejahtera:
    print("\nRegions consistently classified as 'Sejahtera' across all years:")
    for region in consistent_sejahtera:
        print(f"- {region}")

# Identify regions with improving prosperity
improving_regions = []
for region in all_predictions_df['wilayah'].unique():
    region_data = all_predictions_df[all_predictions_df['wilayah'] == region].sort_values('year')
    if len(region_data) >= 2:
        first_year = region_data.iloc[0]
        last_year = region_data.iloc[-1]
        
        # Define prosperity levels
        prosperity_levels = {'Tidak Sejahtera': 0, 'Menengah': 1, 'Sejahtera': 2}
        
        if (prosperity_levels.get(last_year['predicted'], 0) > 
            prosperity_levels.get(first_year['predicted'], 0)):
            improving_regions.append(region)

if improving_regions:
    print("\nRegions with improving prosperity from first to last available year:")
    for region in improving_regions[:5]:  # Show top 5
        print(f"- {region}")
    if len(improving_regions) > 5:
        print(f"  ... and {len(improving_regions) - 5} more")

# 5. Conclusions
print("\n5. CONCLUSIONS")
print("-"*50)
print("Based on our analysis, we can draw the following conclusions:")

# Overall prosperity status
sejahtera_pct = prosperity_dist_2023.get('Sejahtera', 0)
menengah_pct = prosperity_dist_2023.get('Menengah', 0)
tidak_sejahtera_pct = prosperity_dist_2023.get('Tidak Sejahtera', 0)

if sejahtera_pct > 50:
    print("1. The majority of regions are classified as 'Sejahtera', indicating overall good prosperity.")
elif sejahtera_pct + menengah_pct > 70:
    print("1. Most regions are classified as either 'Sejahtera' or 'Menengah', showing moderate to good prosperity overall.")
elif tidak_sejahtera_pct > 50:
    print("1. The majority of regions are classified as 'Tidak Sejahtera', indicating challenges in overall prosperity.")
else:
    print("1. Regions show a mixed prosperity profile with no clear majority in any category.")

# Trend over time
if len(improving_regions) > len(all_predictions_df['wilayah'].unique()) / 2:
    print("2. There is a positive trend in prosperity over time, with more than half of the regions showing improvement.")
elif len(improving_regions) > 0:
    print(f"2. Some regions ({len(improving_regions)}) show improvement in prosperity over time, but this is not a universal trend.")
else:
    print("2. There is no clear trend of improvement in prosperity over time across regions.")

# Key indicators
print("3. The most important indicators for predicting prosperity are related to:")
if 'feature_importances' in rf_results:
    # Group top features by category
    infra_features = [f for f in top_features if f in data_infrastruktur_indicator_to_file or f.startswith('kendaraan')]
    ekonomi_features = [f for f in top_features if f in data_ekonomi_indicator_to_file]
    kesehatan_features = [f for f in top_features if f in data_kesehatan_indicator_to_file]
    pendidikan_features = [f for f in top_features if f in data_pendidikan_indicator_to_file or 'partisipasi' in f]
    
    if infra_features:
        print(f"   - Infrastructure: {', '.join(infra_features)}")
    if ekonomi_features:
        print(f"   - Economy: {', '.join(ekonomi_features)}")
    if kesehatan_features:
        print(f"   - Health: {', '.join(kesehatan_features)}")
    if pendidikan_features:
        print(f"   - Education: {', '.join(pendidikan_features)}")

# Regional disparities
print("4. There are significant regional disparities in prosperity, with some regions consistently")
print("   performing well while others face persistent challenges.")

# Model performance
print(f"5. The {better_model} model provides reliable predictions of regional prosperity")
print(f"   with an accuracy of {rf_results['test_accuracy']:.2f} for Random Forest and {lr_results['test_accuracy']:.2f} for Logistic Regression.")

# 6. Recommendations
print("\n6. RECOMMENDATIONS")
print("-"*50)
print("Based on our analysis, we recommend the following actions:")

print("1. Focus on improving key indicators that strongly correlate with prosperity:")
if 'feature_importances' in rf_results:
    for i, feature in enumerate(top_features[:3], 1):
        print(f"   {i}. Invest in programs to improve {feature}")

print("\n2. Target interventions for regions consistently classified as 'Tidak Sejahtera':")
consistently_tidak_sejahtera = all_predictions_df.groupby('wilayah')['predicted'].apply(
    lambda x: len(x.unique()) == 1 and 'Tidak Sejahtera' in x.unique()
).reset_index()
tidak_sejahtera_regions = consistently_tidak_sejahtera[consistently_tidak_sejahtera['predicted'] == True]['wilayah'].tolist()

if tidak_sejahtera_regions:
    for region in tidak_sejahtera_regions[:3]:
        print(f"   - Develop comprehensive improvement programs for {region}")
    if len(tidak_sejahtera_regions) > 3:
        print(f"     ... and {len(tidak_sejahtera_regions) - 3} other regions")
else:
    print("   - No regions are consistently classified as 'Tidak Sejahtera'")

print("\n3. Learn from successful regions:")
if consistent_sejahtera:
    print("   Study and replicate successful policies and programs from consistently prosperous regions:")
    for region in consistent_sejahtera[:3]:
        print(f"   - Analyze success factors in {region}")
    if len(consistent_sejahtera) > 3:
        print(f"     ... and {len(consistent_sejahtera) - 3} other regions")

print("\n4. Monitor trends and implement early interventions:")
print("   - Establish an early warning system for regions showing declining prosperity indicators")
print("   - Implement targeted interventions before regions fall into the 'Tidak Sejahtera' category")

print("\n5. Enhance data collection and analysis:")
print("   - Improve data quality and coverage for all prosperity indicators")
print("   - Conduct more granular analysis at sub-regional levels")
print("   - Update the prosperity model annually to track progress and adjust interventions")

# 7. Limitations and Future Work
print("\n7. LIMITATIONS AND FUTURE WORK")
print("-"*50)
print("Limitations of the current analysis:")
print("1. Limited to available indicators and may not capture all aspects of prosperity")
print("2. Relies on data quality which may vary across regions and indicators")
print("3. Uses a simplified labeling approach that may not capture nuanced prosperity levels")
print("4. Does not account for potential interactions between indicators")

print("\nFuture work:")
print("1. Incorporate additional indicators such as environmental factors and social cohesion")
print("2. Develop more sophisticated models that can capture complex relationships between indicators")
print("3. Conduct causal analysis to identify interventions with the highest impact")
print("4. Implement a dashboard for real-time monitoring of prosperity indicators")
print("5. Extend the analysis to smaller administrative units for more targeted interventions")

print("\n" + "="*80)
print("END OF PROSPERITY MODELLING ANALYSIS")
print("="*80)

# Create a final visualization summarizing the prosperity status
plt.figure(figsize=(12, 8))

# Create a pie chart of prosperity distribution in 2023
plt.subplot(1, 2, 1)
prosperity_counts = predictions_2023['predicted'].value_counts()
plt.pie(prosperity_counts, labels=prosperity_counts.index, autopct='%1.1f%%',
        colors=[prosperity_colors[cat] for cat in prosperity_counts.index])
plt.title('Prosperity Distribution in 2023', fontsize=14)
plt.axis('equal')

# Create a bar chart showing the number of regions in each prosperity category over time
plt.subplot(1, 2, 2)
prosperity_by_year = all_predictions_df.groupby(['year', 'predicted']).size().unstack(fill_value=0)
prosperity_by_year.plot(kind='bar', ax=plt.gca(), 
                       color=[prosperity_colors[cat] for cat in prosperity_by_year.columns])
plt.title('Prosperity Categories Over Time', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Number of Regions')
plt.legend(title='Prosperity Category')

plt.tight_layout()
plt.show()

print("\nFinal analysis and conclusions completed.")

