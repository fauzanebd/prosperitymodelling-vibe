import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure
import json

def _fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_confusion_matrix_plot(cm):
    """
    Generate a confusion matrix plot
    
    Parameters:
    -----------
    cm : list
        Confusion matrix as a 2D list
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    cm = np.array(cm)
    
    # Calculate percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # Add percentage labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                text = ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                              ha='center', va='center', color='black')
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Set tick labels
    ax.set_xticklabels(['Not Sejahtera', 'Sejahtera'])
    ax.set_yticklabels(['Not Sejahtera', 'Sejahtera'])
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def generate_feature_importance_plot(feature_importance):
    """
    Generate a feature importance plot
    
    Parameters:
    -----------
    feature_importance : dict
        Dictionary mapping feature names to importance values
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    # Convert to DataFrame
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot feature importance
    sns.barplot(x='Importance', y='Feature', data=df, ax=ax)
    
    # Set labels
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def generate_indicator_distribution_plot(df, indicator_name, year=None):
    """
    Generate a distribution plot for an indicator
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the indicator data
    indicator_name : str
        Name of the indicator
    year : str or int, optional
        Year to filter for
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distribution
    sns.histplot(df[indicator_name], kde=True, ax=ax)
    
    # Add vertical lines for quartiles
    q1 = df[indicator_name].quantile(0.25)
    q3 = df[indicator_name].quantile(0.75)
    
    ax.axvline(q1, color='r', linestyle='--', label=f'Q1: {q1:.2f}')
    ax.axvline(q3, color='g', linestyle='--', label=f'Q3: {q3:.2f}')
    
    # Set labels
    ax.set_xlabel(indicator_name)
    ax.set_ylabel('Count')
    title = f'Distribution of {indicator_name}'
    if year:
        title += f' ({year})'
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def generate_indicator_trend_plot(df, indicator_name):
    """
    Generate a trend plot for an indicator
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the indicator data
    indicator_name : str
        Name of the indicator
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    # Group by year and calculate mean
    yearly_mean = df.groupby('year')[indicator_name].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trend
    sns.lineplot(x='year', y=indicator_name, data=yearly_mean, marker='o', ax=ax)
    
    # Set labels
    ax.set_xlabel('Year')
    ax.set_ylabel(indicator_name)
    ax.set_title(f'Trend of {indicator_name} Over Time')
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def generate_regional_comparison_plot(df, indicator_name, year=None, top_n=20):
    """
    Generate a regional comparison plot for an indicator
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the indicator data
    indicator_name : str
        Name of the indicator
    year : str or int, optional
        Year to filter for
    top_n : int, optional
        Number of top regions to show
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    # Sort by indicator value
    df_sorted = df.sort_values(indicator_name, ascending=False)
    
    # Take top N regions
    df_top = df_sorted.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot regional comparison
    bars = sns.barplot(x=indicator_name, y='provinsi', data=df_top, 
                      hue='label_sejahtera', palette={'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'},
                      ax=ax)
    
    # Set labels
    ax.set_xlabel(indicator_name)
    ax.set_ylabel('Region')
    title = f'Top {top_n} Regions by {indicator_name}'
    if year:
        title += f' ({year})'
    ax.set_title(title)
    
    # Add legend
    ax.legend(title='Prosperity Label')
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def generate_prosperity_distribution_plot(df):
    """
    Generate a distribution plot for prosperity predictions
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the prediction data
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    # Count predictions by class
    counts = df['predicted_class'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distribution
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    bars = ax.bar(counts.index, counts.values, color=[colors.get(c, 'blue') for c in counts.index])
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.0f}', ha='center', va='bottom')
    
    # Set labels
    ax.set_xlabel('Prosperity Class')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prosperity Predictions')
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def generate_prosperity_trend_plot(df):
    """
    Generate a trend plot for prosperity predictions
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the prediction data
        
    Returns:
    --------
    str
        Base64-encoded image
    """
    # Group by year and predicted class
    yearly_counts = df.groupby(['year', 'predicted_class']).size().unstack().fillna(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trend
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    for cls in yearly_counts.columns:
        ax.plot(yearly_counts.index, yearly_counts[cls], marker='o', label=cls, color=colors.get(cls, 'blue'))
    
    # Set labels
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Trend of Prosperity Predictions Over Time')
    
    # Add legend
    ax.legend(title='Prosperity Class')
    
    # Convert to base64
    img_str = _fig_to_base64(fig)
    plt.close(fig)
    
    return img_str 