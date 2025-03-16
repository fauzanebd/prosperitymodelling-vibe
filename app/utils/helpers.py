import pandas as pd
import numpy as np
from app.utils.constants import REVERSE_INDICATORS

def format_indicator_name(indicator_name):
    """Format indicator name for display"""
    return indicator_name.replace('_', ' ').title()

def get_all_provinces():
    """Get a list of all provinces in Indonesia"""
    return [
        'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 
        'Sumatera Selatan', 'Bengkulu', 'Lampung', 'Kepulauan Bangka Belitung', 
        'Kepulauan Riau', 'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 
        'DI Yogyakarta', 'Jawa Timur', 'Banten', 'Bali', 'Nusa Tenggara Barat', 
        'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Tengah', 
        'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara', 
        'Sulawesi Utara', 'Sulawesi Tengah', 'Sulawesi Selatan', 
        'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat', 'Maluku', 
        'Maluku Utara', 'Papua Barat', 'Papua'
    ]

def is_reverse_indicator(indicator_name):
    """Check if an indicator is reverse (lower values are better)"""
    return indicator_name in REVERSE_INDICATORS

def calculate_prediction_stats(predictions):
    """Calculate statistics for predictions"""
    if not predictions:
        return None
    
    # Count predictions by class
    prediction_counts = {}
    for prediction in predictions:
        prediction_counts[prediction.predicted_class] = prediction_counts.get(prediction.predicted_class, 0) + 1
    
    # Calculate percentages
    total_predictions = len(predictions)
    prediction_percentages = {cls: (count / total_predictions) * 100 
                             for cls, count in prediction_counts.items()}
    
    # Combine counts and percentages
    prediction_stats = {
        'total': total_predictions,
        'classes': {
            cls: {
                'count': count,
                'percentage': prediction_percentages[cls]
            } for cls, count in prediction_counts.items()
        }
    }
    
    return prediction_stats

def paginate_query(query, page, per_page):
    """Paginate a SQLAlchemy query"""
    return query.paginate(page=page, per_page=per_page, error_out=False) 