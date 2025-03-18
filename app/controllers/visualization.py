from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from app.models.ml_models import TrainedModel
from app.models.predictions import RegionPrediction
from app.models.indicators import INDICATOR_MODELS, IndeksPembangunanManusia
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
    generate_prosperity_comparison_plot,
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
        parameters = selected_model.get_parameters()
        
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
        if metrics.get('confusion_matrix') is not None:
            confusion_matrix_json = generate_confusion_matrix_plot(metrics['confusion_matrix'])
        
        # Generate feature importance plot
        feature_importance_json = None
        if metrics.get('feature_importance'):
            feature_importance_json = generate_feature_importance_plot(metrics['feature_importance'])
        
        # Generate cross-validation scores plot
        cv_scores_json = None
        if metrics.get('cv_scores'):
            import plotly.graph_objects as go
            import numpy as np
            
            # Create figure for CV scores
            fig = go.Figure()
            
            # Add bar chart for CV scores
            cv_scores = metrics['cv_scores']
            fold_numbers = list(range(1, len(cv_scores) + 1))
            
            # Add bars for each fold's accuracy
            fig.add_trace(
                go.Bar(
                    x=fold_numbers,
                    y=[score * 100 for score in cv_scores],
                    name='Fold Accuracy',
                    text=[f"{score * 100:.2f}%" for score in cv_scores],
                    textposition='auto',
                    marker_color='rgb(26, 118, 255)'
                )
            )
            
            # Add a line for the mean accuracy
            mean_accuracy = metrics['mean_cv_accuracy'] * 100
            fig.add_trace(
                go.Scatter(
                    x=fold_numbers,
                    y=[mean_accuracy] * len(fold_numbers),
                    mode='lines',
                    name=f'Mean Accuracy: {mean_accuracy:.2f}%',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
            
            # Add error bars showing standard deviation
            std_accuracy = metrics['std_cv_accuracy'] * 100
            fig.add_trace(
                go.Scatter(
                    x=fold_numbers,
                    y=[mean_accuracy + std_accuracy] * len(fold_numbers),
                    mode='lines',
                    name=f'Mean + Std: {(mean_accuracy + std_accuracy):.2f}%',
                    line=dict(color='green', width=1, dash='dot'),
                    showlegend=True
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=fold_numbers,
                    y=[mean_accuracy - std_accuracy] * len(fold_numbers),
                    mode='lines',
                    name=f'Mean - Std: {(mean_accuracy - std_accuracy):.2f}%',
                    line=dict(color='orange', width=1, dash='dot'),
                    showlegend=True
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Cross-Validation Accuracy by Fold',
                xaxis_title='Fold Number',
                yaxis_title='Accuracy (%)',
                yaxis=dict(range=[
                    max(0, min([score * 100 for score in cv_scores]) - 5),
                    min(100, max([score * 100 for score in cv_scores]) + 5)
                ]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=50, r=50, t=80, b=50),
                plot_bgcolor='rgb(240, 240, 240)'
            )
            
            # Convert to JSON
            cv_scores_json = fig.to_json()
    else:
        metrics = None
        parameters = None
        confusion_matrix_json = None
        feature_importance_json = None
        cv_scores_json = None
    
    return render_template('visualization/model_performance.html',
                          models=models,
                          selected_model=selected_model,
                          model_type=model_type,
                          evaluation_year=evaluation_year,
                          metrics=metrics,
                          parameters=parameters,
                          confusion_matrix_json=confusion_matrix_json,
                          feature_importance_json=feature_importance_json,
                          cv_scores_json=cv_scores_json)

@visualization_bp.route('/visualization/data')
@login_required
def data_visualization():
    """
    Generate data visualizations (EDA)
    """
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    # Get selected visualization type
    viz_type = request.args.get('viz_type', 'distribution')
    
    # Define visualization types
    viz_types = {
        'distribution': 'Indicator Distribution',
        'trend': 'Indicator Trend',
        'label_distribution': 'Label Distribution',
        'label_trend': 'Label Trend',
        'correlation': 'Correlation Matrix',
        'regional_comparison': 'Regional Comparison'
    }
    
    # Get selected indicator
    selected_indicator = request.args.get('indicator', indicators[0])
    
    # Get selected year for distribution, regional, label distribution
    year = request.args.get('year', '2019')
    
    # Get regions with IPM data (training data) - exclude inference-only regions
    regions_with_ipm = db.session.query(IndeksPembangunanManusia.region).distinct().all()
    regions_with_ipm = [r[0] for r in regions_with_ipm]
    
    # Generate plot based on visualization type
    plot_json = None
    
    # For each visualization type, filter data to include only training regions
    if viz_type == 'distribution' and selected_indicator:
        model_class = INDICATOR_MODELS[selected_indicator]
        data = model_class.query.filter(
            model_class.year == int(year),
            model_class.region.in_(regions_with_ipm)
        ).all()
        
        if data:
            values = [d.value for d in data]
            plot_json = generate_indicator_distribution_plot(values, selected_indicator, year)
    
    elif viz_type == 'trend' and selected_indicator:
        model_class = INDICATOR_MODELS[selected_indicator]
        data = model_class.query.filter(
            model_class.region.in_(regions_with_ipm)
        ).all()
        
        if data:
            # Group data by year and calculate mean
            years = sorted(list(set([d.year for d in data])))
            mean_values = []
            for y in years:
                year_data = [d.value for d in data if d.year == y]
                mean_values.append(sum(year_data) / len(year_data) if year_data else 0)
            
            plot_json = generate_indicator_trend_plot(years, mean_values, selected_indicator)
    
    elif viz_type == 'label_distribution' and selected_indicator:
        model_class = INDICATOR_MODELS[selected_indicator]
        data = model_class.query.filter(
            model_class.year == int(year),
            model_class.region.in_(regions_with_ipm)
        ).all()
        
        if data:
            labels = [d.label_sejahtera for d in data]
            plot_json = generate_label_distribution_plot(labels, selected_indicator, year)
    
    elif viz_type == 'label_trend' and selected_indicator:
        model_class = INDICATOR_MODELS[selected_indicator]
        data = model_class.query.filter(
            model_class.region.in_(regions_with_ipm)
        ).all()
        
        if data:
            # Group data by year and count labels
            years = sorted(list(set([d.year for d in data])))
            sejahtera_counts = []
            menengah_counts = []
            tidak_sejahtera_counts = []
            
            for y in years:
                year_data = [d.label_sejahtera for d in data if d.year == y]
                sejahtera_counts.append(year_data.count('Sejahtera'))
                menengah_counts.append(year_data.count('Menengah'))
                tidak_sejahtera_counts.append(year_data.count('Tidak Sejahtera'))
            
            plot_json = generate_label_trend_plot(years, sejahtera_counts, menengah_counts, tidak_sejahtera_counts, selected_indicator)
    
    elif viz_type == 'correlation':
        # Create a combined dataset for the selected year with only training regions
        combined_data = {}
        
        for indicator_name in indicators:
            model_class = INDICATOR_MODELS[indicator_name]
            data = model_class.query.filter(
                model_class.year == int(year),
                model_class.region.in_(regions_with_ipm)
            ).all()
            
            if data:
                combined_data[indicator_name] = {d.region: d.value for d in data}
        
        if combined_data:
            # Convert to DataFrame
            regions = set()
            for indicator, values in combined_data.items():
                regions.update(values.keys())
            
            df = pd.DataFrame(index=list(regions))
            
            for indicator, values in combined_data.items():
                df[indicator] = df.index.map(values)
            
            plot_json = generate_correlation_matrix_plot(df, year)
    
    elif viz_type == 'regional_comparison' and selected_indicator:
        model_class = INDICATOR_MODELS[selected_indicator]
        data = model_class.query.filter(
            model_class.year == int(year),
            model_class.region.in_(regions_with_ipm)
        ).all()
        
        if data:
            regions = [d.region for d in data]
            values = [d.value for d in data]
            labels = [d.label_sejahtera for d in data]
            
            plot_json = generate_regional_comparison_plot(regions, values, labels, selected_indicator, year)
    
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
    """
    Generate model results visualization
    """
    # Get visualization type
    viz_type = request.args.get('viz_type', 'prosperity_distribution')
    
    # Get result year
    result_year = request.args.get('result_year', '2019')
    
    # Get result type
    result_type = request.args.get('result_type', 'predicted')
    
    # Get region (for region_prediction)
    selected_region = request.args.get('region', '')
    
    # Get filter year for summary stats
    filter_year = request.args.get('filter_year', '2019')
    
    # Get prediction type for summary stats
    prediction_type = request.args.get('prediction_type', 'predicted')
    
    # Get available visualization types
    viz_types = {
        'prosperity_distribution': 'Prosperity Distribution',
        'prosperity_trend': 'Prosperity Trend Over Time',
        'prosperity_comparison': 'Predicted vs Actual Comparison',
        'region_prediction': 'Region-specific Prediction'
    }
    
    # Get the best model based on accuracy
    best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
    
    # Get regions with IPM data (training data)
    regions_with_ipm = db.session.query(IndeksPembangunanManusia.region).distinct().all()
    regions_with_ipm = [r[0] for r in regions_with_ipm]
    
    # Process visualizations
    plot_json = None
    indicators_data = None
    actual_ipm_value = None
    actual_ipm_class = None
    predicted_ipm_class = None
    prediction_probability = None
    regions = []
    
    # Get prediction statistics if a model exists
    prediction_stats = None
    if best_model:
        # Get predictions using the best model for training regions only
        query = RegionPrediction.query.filter(
            RegionPrediction.model_id == best_model.id,
            RegionPrediction.region.in_(regions_with_ipm)
        )
        
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
    if viz_type == 'region_prediction':
        # Get unique regions from predictions
        if best_model:
            region_records = db.session.query(RegionPrediction.region).filter_by(model_id=best_model.id).distinct().all()
            regions = [r[0] for r in region_records]
    
    # Generate visualization based on type
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
                plot_json = generate_prosperity_comparison_plot(df)
    
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