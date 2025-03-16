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
    generate_prosperity_trend_plot
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
    
    # Filter models by type
    filtered_models = [m for m in models if m.model_type == model_type]
    
    # Get the latest model of the selected type
    selected_model = filtered_models[0] if filtered_models else None
    
    if selected_model:
        # Get model metrics
        metrics = selected_model.get_metrics()
        
        # Generate confusion matrix plot
        confusion_matrix_plot = None
        if metrics['confusion_matrix']:
            confusion_matrix_plot = generate_confusion_matrix_plot(metrics['confusion_matrix'])
        
        # Generate feature importance plot
        feature_importance_plot = None
        if metrics['feature_importance']:
            feature_importance_plot = generate_feature_importance_plot(metrics['feature_importance'])
    else:
        metrics = None
        confusion_matrix_plot = None
        feature_importance_plot = None
    
    return render_template('visualization/model_performance.html',
                          models=models,
                          selected_model=selected_model,
                          model_type=model_type,
                          metrics=metrics,
                          confusion_matrix_plot=confusion_matrix_plot,
                          feature_importance_plot=feature_importance_plot)

@visualization_bp.route('/visualization/data')
@login_required
def data_visualization():
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    # Get selected indicator
    selected_indicator = request.args.get('indicator', indicators[0] if indicators else None)
    
    # Get selected visualization type
    viz_type = request.args.get('viz_type', 'distribution')
    
    # Get selected year for distribution and regional comparison
    year = request.args.get('year', '2023')
    
    # Get prediction statistics
    latest_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
    prediction_stats = None
    
    if latest_model:
        # Get predictions using the latest model
        predictions = RegionPrediction.query.filter_by(model_id=latest_model.id).all()
        
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
    
    # Generate visualization based on type
    plot = None
    if selected_indicator and viz_type:
        if viz_type == 'distribution':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.filter_by(year=year).all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.provinsi, d.value, d.label_sejahtera) for d in data],
                                 columns=['provinsi', selected_indicator, 'label_sejahtera'])
                
                plot = generate_indicator_distribution_plot(df, selected_indicator, year=year)
        
        elif viz_type == 'trend':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.provinsi, d.year, d.value, d.label_sejahtera) for d in data],
                                 columns=['provinsi', 'year', selected_indicator, 'label_sejahtera'])
                
                plot = generate_indicator_trend_plot(df, selected_indicator)
        
        elif viz_type == 'regional_comparison':
            # Get data for the selected indicator
            model_class = INDICATOR_MODELS[selected_indicator]
            data = model_class.query.filter_by(year=year).all()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame([(d.provinsi, d.value, d.label_sejahtera) for d in data],
                                 columns=['provinsi', selected_indicator, 'label_sejahtera'])
                
                plot = generate_regional_comparison_plot(df, selected_indicator, year=year)
        
        elif viz_type == 'prosperity_distribution':
            if latest_model:
                # Get predictions
                predictions = RegionPrediction.query.filter_by(model_id=latest_model.id).all()
                
                if predictions:
                    # Convert to DataFrame
                    df = pd.DataFrame([(p.provinsi, p.year, p.predicted_class, p.prediction_probability) for p in predictions],
                                     columns=['provinsi', 'year', 'predicted_class', 'probability'])
                    
                    plot = generate_prosperity_distribution_plot(df)
        
        elif viz_type == 'prosperity_trend':
            if latest_model:
                # Get predictions
                predictions = RegionPrediction.query.filter_by(model_id=latest_model.id).all()
                
                if predictions:
                    # Convert to DataFrame
                    df = pd.DataFrame([(p.provinsi, p.year, p.predicted_class, p.prediction_probability) for p in predictions],
                                     columns=['provinsi', 'year', 'predicted_class', 'probability'])
                    
                    plot = generate_prosperity_trend_plot(df)
    
    return render_template('visualization/data_visualization.html',
                          indicators=indicators,
                          selected_indicator=selected_indicator,
                          viz_type=viz_type,
                          year=year,
                          plot=plot,
                          prediction_stats=prediction_stats)

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