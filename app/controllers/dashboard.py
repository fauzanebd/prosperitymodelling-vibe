from flask import Blueprint, render_template, request
from flask_login import login_required
from app.models.predictions import RegionPrediction
from app.models.ml_models import TrainedModel
from app.models.indicators import IndeksPembangunanManusia
from sqlalchemy import func
from app import db

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    # # Get the selected year (default to 'all')
    # selected_year = request.args.get('year', 'all')

    # Get the selected year (default to '2019')
    selected_year = request.args.get('year', '2019')
    
    # Get the best model based on accuracy
    best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
    
    # Get prediction statistics if a model exists
    prediction_stats = None
    if best_model:
        # Get regions with IPM data (training data)
        regions_with_ipm = db.session.query(IndeksPembangunanManusia.region).distinct().all()
        regions_with_ipm = [r[0] for r in regions_with_ipm]
        
        # Get predictions using the best model for training regions only
        query = RegionPrediction.query.filter(
            RegionPrediction.model_id == best_model.id,
            RegionPrediction.region.in_(regions_with_ipm)
        )
        
        # Filter by year if specified
        if selected_year != 'all':
            query = query.filter_by(year=int(selected_year))
        
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
    
    return render_template('dashboard/index.html', 
                          prediction_stats=prediction_stats,
                          best_model=best_model,
                          selected_year=selected_year) 