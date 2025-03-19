import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde

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
            # Determine annotation color based on the value
            annotation_color = "black" if cm[i, j] < np.max(cm) * 0.5 else "white"
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                showarrow=False,
                font=dict(color=annotation_color)
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

def generate_indicator_distribution_plot(values, indicator_name, year):
    """
    Generate a distribution plot for an indicator
    
    Parameters:
    -----------
    values : list
        List of indicator values
    indicator_name : str
        Name of the indicator
    year : str or int
        Year for the data
        
    Returns:
    --------
    str
        JSON string with plotly figure data
    """
    title = f'Distribution of {indicator_name.replace("_", " ").title()} in {year}'
    
    # Create a figure with two y-axes
    fig = go.Figure()
    
    # Calculate histogram bins
    bin_size = (max(values) - min(values)) / 20 if len(values) > 1 else 1
    bins = np.arange(min(values), max(values) + bin_size, bin_size)
    
    # Add histogram (frequency counts)
    fig.add_trace(go.Histogram(
        x=values,
        name='Frequency',
        bingroup=1,
        xbins=dict(
            start=min(values),
            end=max(values),
            size=bin_size
        ),
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add KDE curve
    # Calculate KDE manually
    kde = gaussian_kde(values)
    x_kde = np.linspace(min(values), max(values), 1000)
    y_kde = kde(x_kde)
    
    # Scale the KDE to match the histogram scale - multiply by # of values * bin size
    scaling_factor = len(values) * bin_size
    y_kde_scaled = y_kde * scaling_factor
    
    # Add the KDE curve
    fig.add_trace(go.Scatter(
        x=x_kde,
        y=y_kde_scaled,
        mode='lines',
        name='Distribution',
        line=dict(color='blue', width=2)
    ))
    
    # Calculate quartiles
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    median = np.percentile(values, 50)
    
    # Add vertical lines for quartiles
    fig.add_vline(x=q1, line_dash="dash", line_color="blue", annotation_text=f"Q1: {q1:.2f}")
    fig.add_vline(x=median, line_dash="dash", line_color="green", annotation_text=f"Median: {median:.2f}")
    fig.add_vline(x=q3, line_dash="dash", line_color="red", annotation_text=f"Q3: {q3:.2f}")
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=indicator_name.replace('_', ' ').title(),
        yaxis_title="Frequency",
        template='plotly_white',
        height=500,
        legend_title="Legend",
        bargap=0.1
    )
    
    # Convert to JSON
    return pio.to_json(fig)

def generate_indicator_trend_plot(years, values, indicator_name):
    """
    Generate a trend plot for an indicator over time
    
    Parameters:
    -----------
    years : list
        List of years
    values : list
        List of mean values for each year
    indicator_name : str
        Name of the indicator
        
    Returns:
    --------
    str
        JSON string with plotly figure data
    """
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Year': years,
        'Value': values
    })
    
    # Create line plot
    fig = px.line(
        df, 
        x='Year', 
        y='Value',
        title=f'Trend of {indicator_name.replace("_", " ").title()} Over Time',
        template='plotly_white',
        markers=True
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=indicator_name.replace('_', ' ').title(),
        height=500
    )
    
    # Convert to JSON
    return pio.to_json(fig)

def generate_regional_comparison_plot(regions, values, labels, indicator_name, year):
    """
    Generate a plot comparing regions based on an indicator
    
    Parameters:
    -----------
    regions : list
        List of region names
    values : list
        List of indicator values
    labels : list
        List of labels (Sejahtera, Menengah, etc.)
    indicator_name : str
        Name of the indicator
    year : str or int
        Year for the data
        
    Returns:
    --------
    str
        JSON string with plotly figure data
    """
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Region': regions,
        'Value': values,
        'Label': labels
    })
    
    # Sort by value
    df = df.sort_values('Value')
    
    # Create color mapping for labels
    colors = {
        'Sejahtera': '#28a745',
        'Menengah': '#ffc107',
        'Tidak Sejahtera': '#dc3545'
    }
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Region',
        y='Value',
        color='Label',
        color_discrete_map=colors,
        title=f'Regional Comparison of {indicator_name.replace("_", " ").title()} ({year})',
        template='plotly_white',
        height=600
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Region',
        yaxis_title=indicator_name.replace('_', ' ').title(),
        xaxis={'categoryorder': 'total ascending'}
    )
    
    # Add horizontal line at mean
    mean_value = df['Value'].mean()
    fig.add_shape(
        type='line',
        x0=-0.5,
        y0=mean_value,
        x1=len(regions) - 0.5,
        y1=mean_value,
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
    )
    
    fig.add_annotation(
        x=0,
        y=mean_value,
        text=f'Mean: {mean_value:.2f}',
        showarrow=False,
        yshift=10
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Convert to JSON
    return pio.to_json(fig)

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

def generate_prosperity_trend_plot(df, result_type):
    """
    Generate a line plot showing the trend of prosperity over time
    """
    # Group data by year and count classes
    pivot_df = df.groupby(['year', 'predicted_class']).size().reset_index(name='count')
    
    # Create figure
    fig = go.Figure()
    
    # Add lines for each class
    for class_label in pivot_df['predicted_class'].unique():
        class_data = pivot_df[pivot_df['predicted_class'] == class_label]
        fig.add_trace(
            go.Scatter(
                x=class_data['year'], 
                y=class_data['count'],
                mode='lines+markers',
                name=class_label,
                line=dict(
                    width=3,
                    color='green' if class_label == 'Sejahtera' else 'orange' if class_label == 'Menengah' else 'red'
                ),
                marker=dict(size=10)
            )
        )
    
    # Update layout
    years = sorted(df['year'].unique())
    fig.update_layout(
        title=f"Trend of {'Predicted' if result_type == 'predicted' else 'Actual'} Prosperity Over Time",
        xaxis=dict(
            title='Year',
            tickmode='array',
            tickvals=years,
            ticktext=[str(y) for y in years],
        ),
        yaxis=dict(title='Number of Regions'),
        legend=dict(title='Prosperity Class'),
        hovermode='x unified'
    )
    
    return fig.to_json()

def generate_prosperity_comparison_plot(df):
    """
    Generate a plot comparing predicted vs actual prosperity classes
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Predicted Prosperity", "Actual Prosperity"),
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        horizontal_spacing=0.1
    )
    
    # Count predicted classes
    predicted_counts = df['predicted_class'].value_counts().reset_index()
    predicted_counts.columns = ['class', 'count']
    
    # Count actual classes
    actual_counts = df['actual_class'].value_counts().reset_index()
    actual_counts.columns = ['class', 'count']
    
    # Define colors
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    
    # Add predicted pie chart
    fig.add_trace(
        go.Pie(
            labels=predicted_counts['class'],
            values=predicted_counts['count'],
            name="Predicted",
            marker_colors=[colors.get(cls, 'gray') for cls in predicted_counts['class']],
            textinfo='percent+label',
            hole=0.3
        ),
        row=1, col=1
    )
    
    # Add actual pie chart
    fig.add_trace(
        go.Pie(
            labels=actual_counts['class'],
            values=actual_counts['count'],
            name="Actual",
            marker_colors=[colors.get(cls, 'gray') for cls in actual_counts['class']],
            textinfo='percent+label',
            hole=0.3
        ),
        row=1, col=2
    )
    
    # Create confusion matrix data
    conf_matrix = {}
    for _, row in df.iterrows():
        pred = row['predicted_class']
        actual = row['actual_class']
        key = f"{pred}-{actual}"
        conf_matrix[key] = conf_matrix.get(key, 0) + 1
    
    # Calculate accuracy
    accuracy = (df['predicted_class'] == df['actual_class']).mean() * 100
    
    # Update layout
    fig.update_layout(
        title_text=f"Comparison of Predicted vs Actual Prosperity (Accuracy: {accuracy:.2f}%)",
        height=500,
        margin=dict(t=100, b=50),
        annotations=[
            dict(text=f"Predicted (n={len(df)})", x=0.18, y=0.5, font_size=14, showarrow=False),
            dict(text=f"Actual (n={len(df)})", x=0.82, y=0.5, font_size=14, showarrow=False)
        ]
    )
    
    # Add interactive hover data
    hoverlabel = dict(bgcolor="white", font_size=16, font_family="Arial")
    fig.update_traces(hoverinfo="label+percent+name", hoverlabel=hoverlabel)
    
    return fig.to_json()

def generate_label_distribution_plot(labels, indicator_name, year):
    """
    Generate a pie chart of label distribution for an indicator
    
    Parameters:
    -----------
    labels : list
        List of label values
    indicator_name : str
        Name of the indicator
    year : str or int
        Year for the data
        
    Returns:
    --------
    str
        JSON string with plotly figure data
    """
    # Count each label
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Label': list(label_counts.keys()),
        'Count': list(label_counts.values())
    })
    
    # Create colors based on labels
    colors = {'Sejahtera': '#28a745', 'Menengah': '#ffc107', 'Tidak Sejahtera': '#dc3545'}
    
    # Create pie chart
    fig = px.pie(
        df, 
        names='Label', 
        values='Count',
        title=f'Label Distribution for {indicator_name.replace("_", " ").title()} in {year}',
        color='Label',
        color_discrete_map=colors,
        template='plotly_white'
    )
    
    # Customize layout
    fig.update_layout(
        height=500
    )
    
    # Convert to JSON
    return pio.to_json(fig)

def generate_label_trend_plot(years, sejahtera_counts, menengah_counts, tidak_sejahtera_counts, indicator_name):
    """
    Generate a trend plot of label distribution over time
    
    Parameters:
    -----------
    years : list
        List of years
    sejahtera_counts : list
        List of 'Sejahtera' counts for each year
    menengah_counts : list
        List of 'Menengah' counts for each year
    tidak_sejahtera_counts : list
        List of 'Tidak Sejahtera' counts for each year
    indicator_name : str
        Name of the indicator
        
    Returns:
    --------
    str
        JSON string with plotly figure data
    """
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add bars for each label
    fig.add_trace(go.Bar(
        x=years,
        y=sejahtera_counts,
        name='Sejahtera',
        marker_color='#28a745'
    ))
    
    fig.add_trace(go.Bar(
        x=years,
        y=menengah_counts,
        name='Menengah',
        marker_color='#ffc107'
    ))
    
    fig.add_trace(go.Bar(
        x=years,
        y=tidak_sejahtera_counts,
        name='Tidak Sejahtera',
        marker_color='#dc3545'
    ))
    
    # Customize layout
    fig.update_layout(
        title=f'Label Trend for {indicator_name.replace("_", " ").title()} Over Time',
        xaxis_title='Year',
        yaxis_title='Count',
        barmode='stack',
        template='plotly_white',
        height=500
    )
    
    # Convert to JSON
    return pio.to_json(fig)

def generate_correlation_matrix_plot(df, year):
    """
    Generate a correlation matrix plot for all indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicators as columns and regions as rows
    year : str or int
        Year for the data
        
    Returns:
    --------
    str
        JSON string with plotly figure data
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title=f'Correlation Matrix of Indicators ({year})',
        template='plotly_white',
        height=600,
        width=800
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Indicator',
        yaxis_title='Indicator',
    )
    
    # Convert to JSON
    return pio.to_json(fig)

def generate_comparison_plot(df, region):
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