import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plotly_to_json(fig):
    """Convert a plotly figure to JSON for HTML display"""
    return fig.to_json()

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
        JSON for Plotly figure
    """
    cm = np.array(cm)
    
    # Calculate percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    # Create Plotly figure
    labels = ['Not Sejahtera', 'Sejahtera']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=False
    ))
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                showarrow=False,
                font=dict(color="black")
            )
    
    # Update layout
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        xaxis=dict(side='bottom'),
        width=600,
        height=500
    )
    
    return plotly_to_json(fig)

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
        JSON for Plotly figure
    """
    # Convert to DataFrame
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Create Plotly figure
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        width=800,
        height=600
    )
    
    return plotly_to_json(fig)

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
        JSON for Plotly figure
    """
    # Calculate quartiles
    q1 = df[indicator_name].quantile(0.25)
    q3 = df[indicator_name].quantile(0.75)
    
    # Create Plotly figure
    fig = px.histogram(
        df, 
        x=indicator_name,
        title=f'Distribution of {indicator_name}' + (f' ({year})' if year else ''),
        nbins=20,
        marginal='box'
    )
    
    # Add vertical lines for quartiles
    fig.add_vline(x=q1, line_dash="dash", line_color="red", annotation_text=f"Q1: {q1:.2f}")
    fig.add_vline(x=q3, line_dash="dash", line_color="green", annotation_text=f"Q3: {q3:.2f}")
    
    # Update layout
    fig.update_layout(
        xaxis_title=indicator_name,
        yaxis_title='Count',
        width=800,
        height=500
    )
    
    return plotly_to_json(fig)

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
        JSON for Plotly figure
    """
    # Group by year and calculate mean
    yearly_mean = df.groupby('year')[indicator_name].mean().reset_index()
    
    # Create Plotly figure
    fig = px.line(
        yearly_mean, 
        x='year', 
        y=indicator_name,
        markers=True,
        title=f'Trend of {indicator_name} Over Time'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=indicator_name,
        width=800,
        height=500
    )
    
    return plotly_to_json(fig)

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
        JSON for Plotly figure
    """
    # Sort by indicator value
    df_sorted = df.sort_values(indicator_name, ascending=False)
    
    # Take top N regions
    df_top = df_sorted.head(top_n)
    
    # Create Plotly figure
    fig = px.bar(
        df_top, 
        x=indicator_name, 
        y='provinsi',
        color='label_sejahtera',
        color_discrete_map={'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'},
        title=f'Top {top_n} Regions by {indicator_name}' + (f' ({year})' if year else '')
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=indicator_name,
        yaxis_title='Region',
        width=900,
        height=700
    )
    
    return plotly_to_json(fig)

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
        JSON for Plotly figure
    """
    # Count predictions by class
    counts = df['predicted_class'].value_counts()
    
    # Create Plotly figure
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    fig = px.bar(
        x=counts.index, 
        y=counts.values,
        color=counts.index,
        color_discrete_map=colors,
        title='Distribution of Prosperity Predictions',
        text=counts.values
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Prosperity Class',
        yaxis_title='Count',
        width=700,
        height=500,
        showlegend=False
    )
    
    return plotly_to_json(fig)

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
        JSON for Plotly figure
    """
    # Group by year and predicted class
    yearly_counts = df.groupby(['year', 'predicted_class']).size().unstack().fillna(0)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add lines for each class
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    for cls in yearly_counts.columns:
        fig.add_trace(go.Scatter(
            x=yearly_counts.index, 
            y=yearly_counts[cls],
            mode='lines+markers',
            name=cls,
            line=dict(color=colors.get(cls, 'blue'))
        ))
    
    # Update layout
    fig.update_layout(
        title='Trend of Prosperity Predictions Over Time',
        xaxis_title='Year',
        yaxis_title='Count',
        width=800,
        height=500
    )
    
    return plotly_to_json(fig) 