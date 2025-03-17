from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from app.models.ml_models import TrainedModel
from app.models.predictions import RegionPrediction
from app.models.indicators import INDICATOR_MODELS
from app import db
from app.services.visualization import (
    generate_confusion_matrix_plot,
    generate_feature_importance_plot,
    generate_indicator_distribution_plot,
    generate_indicator_trend_plot,
    generate_regional_comparison_plot,
    generate_prosperity_distribution_plot,
    generate_prosperity_trend_plot,
    generate_label_distribution_plot,
    generate_label_trend_plot,
    generate_correlation_matrix_plot,
)
import json
import pandas as pd
from sqlalchemy import func

visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/visualization/model-performance')
@login_required
def model_performance():
    # Get all trained models
    models = TrainedModel.query.order_by(TrainedModel.created_at.desc()).all()
    
    # Get selected model type
    model_type = request.args.get('model_type', 'random_forest')
    
    # Get selected evaluation year
    evaluation_year = request.args.get('evaluation_year', 'all')
    
    # Filter models by type
    filtered_models = [m for m in models if m.model_type == model_type]
    
    # Get the latest model of the selected type
    selected_model = filtered_models[0] if filtered_models else None
    
    if selected_model:
        # Get model metrics
        metrics = selected_model.get_metrics()
        
        # If a specific year is selected, filter the confusion matrix data
        if evaluation_year != 'all' and metrics:
            # Get predictions for the specific year
            predictions = RegionPrediction.query.filter_by(
                model_id=selected_model.id,
                year=int(evaluation_year)
            ).all()
            
            if predictions:
                # Get actual IPM data for comparison
                from app.models.indicators import INDICATOR_MODELS
                ipm_model = INDICATOR_MODELS.get('indeks_pembangunan_manusia')
                
                if ipm_model:
                    # Create lists to store actual and predicted classes
                    y_true = []
                    y_pred = []
                    
                    # Map class names to numeric values for metrics calculation
                    class_mapping = {'Sejahtera': 1, 'Menengah': 0, 'Tidak Sejahtera': -1}
                    
                    # Collect actual and predicted classes for each region
                    for prediction in predictions:
                        # Get actual IPM data for this region and year
                        ipm_data = ipm_model.query.filter_by(
                            region=prediction.region, 
                            year=int(evaluation_year)
                        ).first()
                        
                        if ipm_data and hasattr(ipm_data, 'label_sejahtera'):
                            # Add to lists if we have both actual and predicted values
                            actual_class = class_mapping.get(ipm_data.label_sejahtera, 0)
                            predicted_class = class_mapping.get(prediction.predicted_class, 0)
                            
                            y_true.append(actual_class)
                            y_pred.append(predicted_class)
                    
                    # Calculate year-specific metrics if we have data
                    if y_true and y_pred:
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                        
                        # Calculate metrics
                        year_accuracy = accuracy_score(y_true, y_pred)
                        
                        # For multi-class metrics, use macro averaging
                        try:
                            year_precision = precision_score(y_true, y_pred, average='macro')
                            year_recall = recall_score(y_true, y_pred, average='macro')
                            year_f1 = f1_score(y_true, y_pred, average='macro')
                        except:
                            # Fallback if there are issues with the metrics calculation
                            year_precision = 0
                            year_recall = 0
                            year_f1 = 0
                        
                        # Calculate confusion matrix
                        year_cm = confusion_matrix(y_true, y_pred)
                        
                        # Update metrics with year-specific values
                        metrics['accuracy'] = year_accuracy
                        metrics['precision'] = year_precision
                        metrics['recall'] = year_recall
                        metrics['f1_score'] = year_f1
                        metrics['confusion_matrix'] = year_cm
        
        # Generate confusion matrix plot
        confusion_matrix_json = None
        if metrics['confusion_matrix'] is not None:
            confusion_matrix_json = generate_confusion_matrix_plot(metrics['confusion_matrix'])
        
        # Generate feature importance plot
        feature_importance_json = None
        if metrics['feature_importance']:
            feature_importance_json = generate_feature_importance_plot(metrics['feature_importance'])
    else:
        metrics = None
        confusion_matrix_json = None
        feature_importance_json = None
    
    return render_template('visualization/model_performance.html',
                          models=models,
                          selected_model=selected_model,
                          model_type=model_type,
                          evaluation_year=evaluation_year,
                          metrics=metrics,
                          confusion_matrix_json=confusion_matrix_json,
                          feature_importance_json=feature_importance_json)

@visualization_bp.route('/visualization/data')
@login_required
def data_visualization():
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    # Get selected visualization type
    viz_type = request.args.get('viz_type', 'distribution')
    
    # Define visualization types for EDA
    viz_types = {
        'distribution': 'Indicator Distribution',
        'trend': 'Trends Over Time',
        'label_distribution': 'Label Distribution',
        'label_trend': 'Label Trend',
        'correlation': 'Correlation Matrix',
        'regional_comparison': 'Regional Comparison'
    }
    
    # Get selected indicator
    selected_indicator = request.args.get('indicator', indicators[0] if indicators else None)
    
    # Get selected year for distribution and regional comparison
    year = request.args.get('year', '2023')
    
    # Generate visualization based on type
    plot_json = None
    
    if selected_indicator:
        if viz_type == 'distribution':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.filter_by(year=year).all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.region, d.value, d.label_sejahtera) for d in data],
                                 columns=['region', selected_indicator, 'label_sejahtera'])
                
                plot_json = generate_indicator_distribution_plot(df, selected_indicator, year=year)
        
        elif viz_type == 'trend':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.region, d.year, d.value, d.label_sejahtera) for d in data],
                                 columns=['region', 'year', selected_indicator, 'label_sejahtera'])
                
                plot_json = generate_indicator_trend_plot(df, selected_indicator)
        
        elif viz_type == 'label_distribution':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.filter_by(year=year).all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.region, d.value, d.label_sejahtera) for d in data],
                                 columns=['region', selected_indicator, 'label_sejahtera'])
                
                plot_json = generate_label_distribution_plot(df, selected_indicator, year=year)
        
        elif viz_type == 'label_trend':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.region, d.year, d.value, d.label_sejahtera) for d in data],
                                 columns=['region', 'year', selected_indicator, 'label_sejahtera'])
                
                plot_json = generate_label_trend_plot(df, selected_indicator)
        
        elif viz_type == 'regional_comparison':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.filter_by(year=year).all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.region, d.value, d.label_sejahtera) for d in data],
                                 columns=['region', selected_indicator, 'label_sejahtera'])
                
                plot_json = generate_regional_comparison_plot(df, selected_indicator, year=year)
        
        elif viz_type == 'correlation':
            # Get data for all indicators for the selected year
            indicator_data = {}
            for indicator in indicators:
                model_class = INDICATOR_MODELS[indicator]
                data = model_class.query.filter_by(year=year).all()
                
                if data:
                    # Convert to list of (region, value) tuples
                    indicator_data[indicator] = {d.region: d.value for d in data}
            
            if indicator_data:
                # Get unique regions
                all_regions = set()
                for indicator, values in indicator_data.items():
                    all_regions.update(values.keys())
                
                # Create DataFrame
                df_data = []
                for region in all_regions:
                    row = {'region': region}
                    for indicator, values in indicator_data.items():
                        row[indicator] = values.get(region, None)
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df = df.set_index('region')
                
                # Drop columns with too many missing values
                df = df.dropna(axis=1, thresh=len(df) * 0.5)
                
                # Fill remaining missing values with mean
                df = df.fillna(df.mean())
                
                plot_json = generate_correlation_matrix_plot(df, year=year)
    
    return render_template('visualization/data_visualization.html',
                          indicators=indicators,
                          selected_indicator=selected_indicator,
                          viz_type=viz_type,
                          viz_types=viz_types,
                          year=year,
                          plot_json=plot_json)

@visualization_bp.route('/visualization/model-results')
@login_required
def model_results_visualization():
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    # Get selected visualization type
    viz_type = request.args.get('viz_type', 'prosperity_distribution')
    
    # Define visualization types for model results
    viz_types = {
        'prosperity_distribution': 'Prosperity Distribution',
        'prosperity_trend': 'Prosperity Trend',
        'prosperity_comparison': 'Prosperity Comparison',
        'region_prediction': 'Region Prediction Analysis'
    }
    
    # Get selected year
    result_year = request.args.get('result_year', '2019')
    
    # Get selected region for region prediction analysis
    selected_region = request.args.get('region', None)
    
    # Get result type
    result_type = request.args.get('result_type', 'predicted')
    
    # Get prediction type and filter year for the top stats card
    prediction_type = request.args.get('prediction_type', 'predicted')
    filter_year = request.args.get('filter_year', '2019')
    
    # Get prediction statistics
    best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
    prediction_stats = None
    
    if best_model:
        # Get predictions using the latest model
        query = RegionPrediction.query.filter_by(model_id=best_model.id)
        
        # Filter by year if specified
        if filter_year != 'all':
            query = query.filter_by(year=int(filter_year))
        
        predictions = query.all()
        
        if predictions:
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
    
    # Get unique regions for dropdown
    regions = []
    if viz_type == 'region_prediction':
        # Get unique regions from predictions
        if best_model:
            region_records = db.session.query(RegionPrediction.region).filter_by(model_id=best_model.id).distinct().all()
            regions = [r[0] for r in region_records]
    
    # Generate visualization based on type
    plot_json = None

    # Initialize variables for template
    actual_ipm_value = None
    actual_ipm_class = None
    predicted_ipm_class = None
    prediction_probability = None
    indicators_data = []
    
    if viz_type == 'prosperity_distribution' and best_model:
        # Get predictions
        query = RegionPrediction.query.filter_by(model_id=best_model.id)
        
        # Filter by year if specified
        if result_year != 'all':
            query = query.filter_by(year=int(result_year))
            
        predictions = query.all()
        
        if predictions:
            # Convert to DataFrame
            df = pd.DataFrame([(p.region, p.year, p.predicted_class, p.prediction_probability) for p in predictions],
                             columns=['region', 'year', 'predicted_class', 'probability'])
            
            plot_json = generate_prosperity_distribution_plot(df, result_type)
    
    elif viz_type == 'prosperity_trend' and best_model:
        # Get predictions
        predictions = RegionPrediction.query.filter_by(model_id=best_model.id).all()
        
        if predictions:
            # Convert to DataFrame
            df = pd.DataFrame([(p.region, p.year, p.predicted_class, p.prediction_probability) for p in predictions],
                             columns=['region', 'year', 'predicted_class', 'probability'])
            
            plot_json = generate_prosperity_trend_plot(df, result_type)
            
    elif viz_type == 'prosperity_comparison' and best_model:
        # Get predictions
        query = RegionPrediction.query.filter_by(model_id=best_model.id)
        
        # Filter by year if specified
        if result_year != 'all':
            query = query.filter_by(year=int(result_year))
            
        predictions = query.all()
        
        if predictions:
            # Get actual IPM data
            ipm_model = INDICATOR_MODELS.get('indeks_pembangunan_manusia')
            actual_data = []
            
            if ipm_model:
                for p in predictions:
                    ipm_data = ipm_model.query.filter_by(region=p.region, year=p.year).first()
                    if ipm_data:
                        actual_data.append({
                            'region': p.region,
                            'year': p.year,
                            'predicted_class': p.predicted_class,
                            'prediction_probability': p.prediction_probability,
                            'actual_class': ipm_data.label_sejahtera
                        })
            
            if actual_data:
                df = pd.DataFrame(actual_data)
                # This would need a new visualization function
                # plot_json = generate_prosperity_comparison_plot(df)
    
    elif viz_type == 'region_prediction' and best_model and selected_region:
        # Get predictions for the selected region
        predictions = RegionPrediction.query.filter_by(
            model_id=best_model.id, 
            region=selected_region,
            year=int(result_year)
        ).first()
        
        # Initialize variables for template
        actual_ipm_value = None
        actual_ipm_class = None
        predicted_ipm_class = None
        prediction_probability = None
        indicators_data = []
        plot_json = None
        
        if predictions:
            predicted_ipm_class = predictions.predicted_class
            prediction_probability = predictions.prediction_probability
            
            # Get actual IPM data if available
            ipm_model = INDICATOR_MODELS.get('indeks_pembangunan_manusia')
            if ipm_model:
                ipm_data = ipm_model.query.filter_by(
                    region=selected_region, 
                    year=int(result_year)
                ).first()
                
                if ipm_data:
                    actual_ipm_value = ipm_data.value
                    actual_ipm_class = ipm_data.label_sejahtera
            
            # Get indicator values used for this prediction
            indicators_data = []
            if best_model.feature_importance:
                feature_importance = best_model.feature_importance
                
                # Check if feature_importance is a dictionary (could be a string for some models like logistic regression)
                if isinstance(feature_importance, dict):
                    # Get top 6 important features
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:6]
                    
                    for feature_name, importance in top_features:
                        if feature_name in INDICATOR_MODELS:
                            indicator_model = INDICATOR_MODELS[feature_name]
                            indicator_data = indicator_model.query.filter_by(
                                region=selected_region,
                                year=int(result_year)
                            ).first()
                            
                            if indicator_data:
                                indicators_data.append({
                                    'name': feature_name,
                                    'value': indicator_data.value,
                                    'importance': importance
                                })
                else:
                    # For models without proper feature importance (like logistic regression in some cases)
                    print(f"Feature importance is not a dictionary: {type(feature_importance)}")
            
            # Generate historical trend plot for this region
            historical_predictions = RegionPrediction.query.filter_by(
                model_id=best_model.id,
                region=selected_region
            ).order_by(RegionPrediction.year).all()
            
            if historical_predictions:
                # Get historical actual data too
                historical_actual = []
                if ipm_model:
                    for pred in historical_predictions:
                        actual = ipm_model.query.filter_by(
                            region=selected_region,
                            year=pred.year
                        ).first()
                        if actual:
                            historical_actual.append({
                                'year': actual.year,
                                'value': actual.value,
                                'label': actual.label_sejahtera
                            })
            
            # Create data for plotting
            import plotly.graph_objects as go
            
            # Create figure
            fig = go.Figure()
            
            # Add IPM value trend line
            if historical_actual:
                years = [item['year'] for item in historical_actual]
                values = [item['value'] for item in historical_actual]
                
                # Add IPM value trend line
                fig.add_trace(
                    go.Scatter(
                        x=years, 
                        y=values,
                        mode='lines+markers',
                        name='IPM Value',
                        line=dict(color='blue', width=3),
                        marker=dict(size=10)
                    )
                )
                
                # Add threshold lines for IPM classification
                # Assuming thresholds for classification (these should match your actual thresholds)
                # Adjust these values based on your actual classification thresholds
                sejahtera_threshold = 70.0
                menengah_threshold = 60.0
                
                fig.add_shape(
                    type="line",
                    x0=min(years),
                    x1=max(years),
                    y0=sejahtera_threshold,
                    y1=sejahtera_threshold,
                    line=dict(color="green", width=2, dash="dash"),
                )
                
                fig.add_shape(
                    type="line",
                    x0=min(years),
                    x1=max(years),
                    y0=menengah_threshold,
                    y1=menengah_threshold,
                    line=dict(color="orange", width=2, dash="dash"),
                )
                
                # Add annotations for the thresholds
                fig.add_annotation(
                    x=max(years),
                    y=sejahtera_threshold,
                    text="Sejahtera Threshold",
                    showarrow=False,
                    xshift=10,
                    font=dict(color="green"),
                )
                
                fig.add_annotation(
                    x=max(years),
                    y=menengah_threshold,
                    text="Menengah Threshold",
                    showarrow=False,
                    xshift=10,
                    font=dict(color="orange"),
                )
                
                # Add labels for the classification at each point
                for i, item in enumerate(historical_actual):
                    label_color = "green" if item['label'] == 'Sejahtera' else "orange" if item['label'] == 'Menengah' else "red"
                    
                    fig.add_annotation(
                        x=item['year'],
                        y=item['value'],
                        text=item['label'],
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=label_color,
                        font=dict(color=label_color),
                        bgcolor="white",
                        bordercolor=label_color,
                        borderwidth=1,
                        borderpad=4,
                        yshift=20
                    )
                
                # Customize the layout
                min_y = min(values) * 0.95
                max_y = max(values) * 1.05
                
                fig.update_layout(
                    title=f'Historical IPM Trend for {selected_region}',
                    xaxis=dict(
                        title='Year',
                        tickmode='array',
                        tickvals=years,
                        ticktext=[str(y) for y in years],
                    ),
                    yaxis=dict(
                        title='IPM Value',
                        range=[min_y, max_y]
                    ),
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(l=20, r=20, t=50, b=20),
                    hovermode='x unified'
                )
                
                # Convert the figure to JSON for the template
                plot_json = fig.to_json()
    
    return render_template('visualization/model_results_visualization.html',
                          viz_type=viz_type,
                          viz_types=viz_types,
                          regions=regions,
                          selected_region=selected_region,
                          actual_ipm_value=actual_ipm_value,
                          actual_ipm_class=actual_ipm_class,
                          predicted_ipm_class=predicted_ipm_class,
                          prediction_probability=prediction_probability,
                          indicators_data=indicators_data,
                          plot_json=plot_json,
                          prediction_stats=prediction_stats,
                          best_model=best_model,
                          prediction_type=prediction_type,
                          filter_year=filter_year,
                          result_type=result_type,
                          result_year=result_year)

@visualization_bp.route('/visualization/api/model/<int:model_id>')
@login_required
def get_model_data(model_id):
    """API endpoint to get model data"""
    model = TrainedModel.query.get_or_404(model_id)
    metrics = model.get_metrics()
    
    return jsonify({
        'id': model.id,
        'model_type': model.model_type,
        'created_at': model.created_at.isoformat(),
        'metrics': metrics,
        'parameters': model.get_parameters()
    }) 