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
    cm : list or numpy.ndarray
        Confusion matrix as a 2D list or array
        
    Returns:
    --------
    str
        JSON for Plotly figure
    """
    cm = np.array(cm)
    
    # Calculate percentages
    # Handle division by zero by using a small epsilon where row sum is zero
    row_sums = cm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1e-10, row_sums)  # Replace zeros with small value
    cm_norm = cm.astype('float') / row_sums[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    # Determine labels based on matrix size
    if cm.shape[0] == 2:  # Binary classification
        labels = ['Menengah', 'Sejahtera']
    elif cm.shape[0] == 3:  # Three classes
        labels = ['Tidak Sejahtera', 'Menengah', 'Sejahtera']
    else:  # Handle other sizes
        labels = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create Plotly figure
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
    
    # Create Plotly figure with histogram and KDE
    fig = px.histogram(
        df, 
        x=indicator_name,
        title=f'Distribution of {indicator_name}' + (f' ({year})' if year else ''),
        nbins=20,
        marginal='violin',  # Use violin plot for KDE-like visualization
        color_discrete_sequence=['#636EFA']
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
    # Group by year and calculate mean, min, max
    yearly_stats = df.groupby('year')[indicator_name].agg(['mean', 'min', 'max']).reset_index()
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add mean line with scatter points
    fig.add_trace(go.Scatter(
        x=yearly_stats['year'], 
        y=yearly_stats['mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add range area
    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['max'],
        mode='lines',
        name='Max',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['min'],
        mode='lines',
        name='Min',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Trend of {indicator_name} Over Time',
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
        y='region',
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

def generate_prosperity_distribution_plot(df, result_type='predicted'):
    """
    Generate a distribution plot for prosperity predictions
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the prediction data
    result_type : str
        Type of result to display ('predicted' or 'actual')
        
    Returns:
    --------
    str
        JSON for Plotly figure
    """
    # Count predictions by class
    counts = df['predicted_class'].value_counts()
    
    # Create title based on result type
    title = f"Distribution of {'Predicted' if result_type == 'predicted' else 'Actual'} Prosperity"
    
    # Create Plotly figure
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    fig = px.bar(
        x=counts.index, 
        y=counts.values,
        color=counts.index,
        color_discrete_map=colors,
        title=title,
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

def generate_prosperity_trend_plot(df, result_type='predicted'):
    """
    Generate a trend plot for prosperity predictions
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the prediction data
    result_type : str
        Type of result to display ('predicted' or 'actual')
        
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
    
    # Update layout with result type in title
    fig.update_layout(
        title=f'Trend of {"Predicted" if result_type == "predicted" else "Actual"} Prosperity Over Time',
        xaxis_title='Year',
        yaxis_title='Count',
        width=800,
        height=500
    )
    
    return plotly_to_json(fig)

def generate_label_distribution_plot(df, indicator_name, year=None):
    """
    Generate a distribution plot for labels of an indicator
    
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
    # Count labels
    label_counts = df['label_sejahtera'].value_counts()
    
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=["Label Distribution", "Label Percentage"]
    )
    
    # Add bar chart
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    fig.add_trace(
        go.Bar(
            x=label_counts.index,
            y=label_counts.values,
            text=label_counts.values,
            textposition='auto',
            marker_color=[colors.get(label, 'blue') for label in label_counts.index]
        ),
        row=1, col=1
    )
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=label_counts.index,
            values=label_counts.values,
            marker=dict(colors=[colors.get(label, 'blue') for label in label_counts.index])
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Label Distribution for {indicator_name}' + (f' ({year})' if year else ''),
        width=1000,
        height=500,
        showlegend=False
    )
    
    return plotly_to_json(fig)

def generate_label_trend_plot(df, indicator_name):
    """
    Generate a trend plot for labels of an indicator
    
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
    # Group by year and label
    yearly_counts = df.groupby(['year', 'label_sejahtera']).size().unstack().fillna(0)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add lines for each label
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    for label in yearly_counts.columns:
        fig.add_trace(go.Scatter(
            x=yearly_counts.index, 
            y=yearly_counts[label],
            mode='lines+markers',
            name=label,
            line=dict(color=colors.get(label, 'blue'))
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Label Trend for {indicator_name} Over Time',
        xaxis_title='Year',
        yaxis_title='Count',
        width=800,
        height=500
    )
    
    return plotly_to_json(fig)

def generate_correlation_matrix_plot(df, year=None):
    """
    Generate a correlation matrix plot for all indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all indicator data
    year : str or int, optional
        Year to filter for
        
    Returns:
    --------
    str
        JSON for Plotly figure
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create Plotly figure
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f'Correlation Matrix of Indicators' + (f' ({year})' if year else '')
    )
    
    # Update layout
    fig.update_layout(
        width=900,
        height=800
    )
    
    return plotly_to_json(fig)

    """
    Generate a comparison plot for a specific region's predictions vs actual
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing prediction and actual data
    region : str
        Name of the region to analyze
        
    Returns:
    --------
    str
        JSON for Plotly figure
    """
    # Filter for the selected region
    region_df = df[df['region'] == region]
    
    if region_df.empty:
        return None
    
    # Create Plotly figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["Actual IPM Class", "Predicted IPM Class"]
    )
    
    # Add actual indicator
    actual_class = region_df['actual_class'].iloc[0] if 'actual_class' in region_df.columns else "Unknown"
    actual_color = 'green' if actual_class == 'Sejahtera' else 'orange' if actual_class == 'Menengah' else 'red'
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=region_df['actual_ipm'].iloc[0] if 'actual_ipm' in region_df.columns else 0,
            title={"text": "Actual IPM"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': actual_color},
                'steps': [
                    {'range': [0, 60], 'color': 'lightgray'},
                    {'range': [60, 70], 'color': 'gray'},
                    {'range': [70, 100], 'color': 'darkgray'}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Add predicted indicator
    predicted_class = region_df['predicted_class'].iloc[0]
    predicted_color = 'green' if predicted_class == 'Sejahtera' else 'orange' if predicted_class == 'Menengah' else 'red'
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=region_df['probability'].iloc[0] * 100,  # Convert to percentage
            title={"text": "Prediction Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': predicted_color},
                'steps': [
                    {'range': [0, 33], 'color': 'lightgray'},
                    {'range': [33, 66], 'color': 'gray'},
                    {'range': [66, 100], 'color': 'darkgray'}
                ]
            }
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Prosperity Analysis for {region}',
        width=900,
        height=400
    )
    
    return plotly_to_json(fig) 