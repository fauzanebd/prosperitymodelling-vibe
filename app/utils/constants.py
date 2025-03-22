# Prosperity labels
PROSPERITY_LABELS = {
    'Sejahtera': 'Sejahtera',
    'Menengah': 'Menengah',
    'Tidak Sejahtera': 'Tidak Sejahtera'
}

# Years range for data
YEARS_RANGE = range(2019, 2024)

# Indicators with reverse labeling (lower values are better)
REVERSE_INDICATORS = [
    'tingkat_pengangguran_terbuka',
    'penduduk_miskin',
    'kematian_balita',
    'kematian_bayi',
    'kematian_ibu',
    'persentase_balita_stunting'
]

# Model types
MODEL_TYPES = {
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression'
}

# Visualization types
VISUALIZATION_TYPES = {
    'distribution': 'Distribusi',
    'trend': 'Trend',
    'regional_comparison': 'Perbandingan Regional',
    'prosperity_distribution': 'Distribusi Kesejahteraan',
    'prosperity_trend': 'Trend Kesejahteraan'
}

# Pagination
ITEMS_PER_PAGE = 20 