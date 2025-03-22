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
    
    # Format feature names for better readability
    df['Feature_Formatted'] = df['Feature'].apply(
        lambda x: ' '.join(word.capitalize() for word in x.split('_'))
    )

    # drop year from feature_formatted
    df.drop(df.loc[df['Feature'] == 'year'].index, inplace=True)
    
    # Create Plotly figure
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature_Formatted',
        orientation='h',
        title='Feature Importance'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Garis Horizontal',
        yaxis_title='Garis Vertikal',
        width=600,  # Adjust width to fit card better
        height=600,
        margin=dict(l=200, r=20, t=30, b=50),  # Increase left margin for labels
        autosize=True
    )
    
    # Tilt the y-axis labels to make them fit better
    fig.update_yaxes(tickangle=-30, automargin=True)
    
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
        title=f'Perbandingan Wilayah untuk indikator {indicator_name.replace("_", " ").title()} ({year})',
        template='plotly_white',
        height=600
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Wilayah',
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
    Generate a distribution plot for prosperity predictions vs actual
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the prediction data with both predicted and actual classes
        
    Returns:
    --------
    str
        JSON for Plotly figure
    """
    # Get actual IPM data if not already in the dataframe
    if 'actual_class' not in df.columns:
        # Try to get actual data from region and year
        if 'region' in df.columns and 'year' in df.columns:
            # This part would need to be implemented by fetching actual data
            # For now, we'll just show the predicted data
            actual_data_available = False
        else:
            actual_data_available = False
    else:
        actual_data_available = True
    
    # Count predictions by class
    pred_counts = df['predicted_class'].value_counts().reset_index()
    pred_counts.columns = ['Kelas Kesejahteraan', 'count']
    pred_counts['type'] = 'Prediksi'
    
    # If actual data is available, count by class
    if actual_data_available:
        actual_counts = df['actual_class'].value_counts().reset_index()
        actual_counts.columns = ['Kelas Kesejahteraan', 'count']
        actual_counts['type'] = 'Aktual'
        
        # Combine the data
        combined_data = pd.concat([pred_counts, actual_counts])
    else:
        combined_data = pred_counts
    
    # Define colors
    colors = {'Sejahtera': '#28a745', 'Menengah': '#ffc107', 'Tidak Sejahtera': '#dc3545'}
    
    # Create Plotly figure for grouped bar chart
    fig = px.bar(
        combined_data,
        x='Kelas Kesejahteraan',
        y='count',
        color='Kelas Kesejahteraan',
        barmode='group',
        facet_col='type',
        color_discrete_map=colors,
        text='count',
        title="Perbandingan Distribusi Kesejahteraan Prediksi vs Aktual"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Kelas Kesejahteraan',
        yaxis_title='Jumlah Wilayah',
        width=900,
        height=500,
        legend_title="Kelas Kesejahteraan",
    )
    
    # Clean up the facet labels
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    return plotly_to_json(fig)

def generate_prosperity_trend_plot(df, result_type=None):
    """
    Generate a line plot showing the trend of prosperity over time for both predicted and actual
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the prediction data with both predicted and actual classes
    result_type : str, optional
        Not used anymore, kept for backward compatibility
        
    Returns:
    --------
    str
        JSON for Plotly figure
    """
    # Check if we have actual data
    has_actual_data = 'actual_class' in df.columns
    
    # Create figure
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=("Kesejahteraan (Prediksi)", "Kesejahteraan (Aktual)"),
                       shared_yaxes=True)
    
    # Colors for classes
    colors = {
        'Sejahtera': '#28a745', 
        'Menengah': '#ffc107', 
        'Tidak Sejahtera': '#dc3545'
    }
    
    # Process predicted data
    # Group data by year and count classes
    pred_pivot = df.groupby(['year', 'predicted_class']).size().reset_index(name='count')
    
    # Add lines for each predicted class
    for class_label in pred_pivot['predicted_class'].unique():
        class_data = pred_pivot[pred_pivot['predicted_class'] == class_label]
        fig.add_trace(
            go.Scatter(
                x=class_data['year'], 
                y=class_data['count'],
                mode='lines+markers',
                name=f"Prediksi: {class_label}",
                line=dict(
                    width=3,
                    color=colors.get(class_label, '#000000')
                ),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # Process actual data if available
    if has_actual_data:
        actual_pivot = df.groupby(['year', 'actual_class']).size().reset_index(name='count')
        
        # Add lines for each actual class
        for class_label in actual_pivot['actual_class'].unique():
            class_data = actual_pivot[actual_pivot['actual_class'] == class_label]
            fig.add_trace(
                go.Scatter(
                    x=class_data['year'], 
                    y=class_data['count'],
                    mode='lines+markers',
                    name=f"Aktual: {class_label}",
                    line=dict(
                        width=3,
                        color=colors.get(class_label, '#000000'),
                        dash='dot'  # Dotted line to distinguish from prediction
                    ),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )
    
    # Get unique years
    years = sorted(df['year'].unique())
    
    # Update layout
    fig.update_layout(
        title="Perbandingan Tren Kesejahteraan Prediksi vs Aktual",
        height=500,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(
        title='Tahun',
        tickmode='array',
        tickvals=years,
        ticktext=[str(y) for y in years],
    )
    
    fig.update_yaxes(title='Jumlah Wilayah')
    
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
    predicted_counts.columns = ['Kelas Kesejahteraan', 'count']
    
    # Count actual classes
    actual_counts = df['actual_class'].value_counts().reset_index()
    actual_counts.columns = ['Kelas Kesejahteraan', 'count']
    
    # Define colors
    colors = {'Sejahtera': 'green', 'Menengah': 'orange', 'Tidak Sejahtera': 'red'}
    
    # Add predicted pie chart
    fig.add_trace(
        go.Pie(
            labels=predicted_counts['Kelas Kesejahteraan'],
            values=predicted_counts['count'],
            name="Predicted",
            marker_colors=[colors.get(cls, 'gray') for cls in predicted_counts['Kelas Kesejahteraan']],
            textinfo='percent+label',
            hole=0.3
        ),
        row=1, col=1
    )
    
    # Add actual pie chart
    fig.add_trace(
        go.Pie(
            labels=actual_counts['Kelas Kesejahteraan'],
            values=actual_counts['count'],
            name="Actual",
            marker_colors=[colors.get(cls, 'gray') for cls in actual_counts['Kelas Kesejahteraan']],
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
        title=f'Matriks Korelasi Indikator ({year})',
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
        title=f'Sejahteraku for {region}',
        width=900,
        height=400
    )
    
    return plotly_to_json(fig) 