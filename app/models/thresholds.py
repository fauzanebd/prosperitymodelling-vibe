from app import db

class LabelingThreshold(db.Model):
    """Model for storing labeling thresholds for indicators"""
    __tablename__ = 'labeling_thresholds'
    
    id = db.Column(db.Integer, primary_key=True)
    indicator = db.Column(db.String(255), unique=True, nullable=False)
    sejahtera_threshold = db.Column(db.String(255), nullable=True)
    menengah_threshold = db.Column(db.String(255), nullable=True)
    tidak_sejahtera_threshold = db.Column(db.String(255), nullable=True)
    labeling_method = db.Column(db.String(50), default='IQR') # 'iqr', 'manual', or other methods
    is_reverse = db.Column(db.Boolean, default=False)  # True if lower values are better (e.g., unemployment rate)
    low_threshold = db.Column(db.Float, nullable=True)
    high_threshold = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f'<LabelingThreshold {self.indicator}>'
 