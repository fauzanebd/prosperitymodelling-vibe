from flask import Blueprint, render_template
from flask_login import login_required
from app.models.predictions import RegionPrediction
from app.models.ml_models import TrainedModel
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    # Get the latest model
    latest_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
    
    # Get prediction statistics if a model exists
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
    
    return render_template('dashboard/index.html', 
                          prediction_stats=prediction_stats,
                          latest_model=latest_model) 