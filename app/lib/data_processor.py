import pandas as pd
import numpy as np
from app.models.thresholds import LabelingThreshold

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

    # Delete region data
    df_melted = df_melted[~df_melted['wilayah'].str.contains('region')]
    
    # Convert values to numeric
    df_melted[indicator_name] = pd.to_numeric(df_melted[indicator_name], errors='coerce')

    # Convert year to int
    df_melted['year'] = df_melted['year'].astype(int)
    
    return df_melted

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
        # If no match found, use the third column
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

def label_iqr(indicator_name, df, column, reverse=False):
    """
    Function to determine categories based on IQR.
    If reverse=True, then lower values are better (e.g., Penduduk Miskin).
    If reverse=False, then higher values are better (e.g., Upah Minimum, PDRB).
    
    Parameters:
    -----------
    indicator_name : str
        Name of the indicator
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
    models.LabelingThreshold
        Threshold data
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    low_threshold = Q1  
    high_threshold = Q3
    
    # Convert NumPy types to Python native types
    if isinstance(low_threshold, np.integer):
        low_threshold = int(low_threshold)
    elif isinstance(low_threshold, np.floating):
        low_threshold = float(low_threshold)
        
    if isinstance(high_threshold, np.integer):
        high_threshold = int(high_threshold)
    elif isinstance(high_threshold, np.floating):
        high_threshold = float(high_threshold)

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
            
    threshold_data = {
        'indicator': indicator_name,
        'sejahtera_threshold': f"< {low_threshold}" if reverse else f"> {high_threshold}",
        'menengah_threshold': f"{low_threshold} - {high_threshold}",
        'tidak_sejahtera_threshold': f"> {high_threshold}" if reverse else f"< {low_threshold}",
        'labeling_method': 'IQR',
        'is_reverse': reverse,
        'low_threshold': float(low_threshold),
        'high_threshold': float(high_threshold)
    }
    threshold = LabelingThreshold(**threshold_data)
    
    return df[column].apply(categorize), threshold