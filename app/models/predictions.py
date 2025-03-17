from app import db
from datetime import datetime

class RegionPrediction(db.Model):
    __tablename__ = 'region_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(64), index=True)
    year = db.Column(db.Integer, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey('trained_models.id'))
    predicted_class = db.Column(db.String(64))  # 'Sejahtera', 'Menengah', 'Tidak Sejahtera'
    prediction_probability = db.Column(db.Float)  # Probability of the predicted class
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Add unique constraint to ensure we don't have duplicate predictions
    __table_args__ = (
        db.UniqueConstraint('region', 'year', 'model_id', name='uix_region_prediction'),
    )
    
    def __repr__(self):
        return f'<RegionPrediction {self.region} {self.year} {self.predicted_class}>' 