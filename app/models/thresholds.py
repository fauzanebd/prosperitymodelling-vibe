from app import db

class LabelingThreshold(db.Model):
    """Model for storing labeling thresholds for indicators"""
    __tablename__ = 'labeling_thresholds'
    
    id = db.Column(db.Integer, primary_key=True)
    indicator = db.Column(db.String(255), unique=True, nullable=False)
    sejahtera_threshold = db.Column(db.String(255), nullable=True)
    menengah_threshold = db.Column(db.String(255), nullable=True)
    tidak_sejahtera_threshold = db.Column(db.String(255), nullable=True)
    labeling_method = db.Column(db.String(50), default='iqr') # 'iqr', 'manual', or other methods
    is_reverse = db.Column(db.Boolean, default=False)  # True if lower values are better (e.g., unemployment rate)
    
    def __repr__(self):
        return f'<LabelingThreshold {self.indicator}>'
    
    # @staticmethod
    # def init_default_thresholds():
    #     """Initialize default thresholds for all indicators"""
    #     # Check if entries already exist
    #     if LabelingThreshold.query.count() > 0:
    #         return
            
        # # Import indicator models
        # from app.models.indicators import INDICATOR_MODELS
            
        # # Define default thresholds for manual labeling
        # manual_thresholds = {
        #     'indeks_pembangunan_manusia': {
        #         'sejahtera_threshold': 70,
        #         'menengah_threshold': 60,
        #         'labeling_method': 'manual',
        #         'is_reverse': False
        #     },
        #     'tingkat_pengangguran_terbuka': {
        #         'sejahtera_threshold': 6.75,
        #         'menengah_threshold': 7.0,
        #         'labeling_method': 'manual',
        #         'is_reverse': True
        #     },
        #     'persentase_balita_stunting': {
        #         'sejahtera_threshold': 20,
        #         'menengah_threshold': 29,
        #         'labeling_method': 'manual',
        #         'is_reverse': True
        #     }
        # }
        
        # # Define indicators where lower values are better (reverse)
        # reverse_indicators = [
        #     'penduduk_miskin',
        #     'kematian_balita',
        #     'kematian_bayi',
        #     'kematian_ibu'
        # ]
        
        # # Add all indicators
        # for indicator in INDICATOR_MODELS.keys():
        #     # Check if it's a manually labeled indicator
        #     if indicator in manual_thresholds:
        #         threshold_data = {
        #             'indicator': indicator,
        #             'sejahtera_threshold': manual_thresholds[indicator]['sejahtera_threshold'],
        #             'menengah_threshold': manual_thresholds[indicator]['menengah_threshold'],
        #             'tidak_sejahtera_threshold': None,
        #             'labeling_method': manual_thresholds[indicator]['labeling_method'],
        #             'is_reverse': manual_thresholds[indicator]['is_reverse']
        #         }
        #     else:
        #         # Default to IQR method
        #         threshold_data = {
        #             'indicator': indicator,
        #             'sejahtera_threshold': None,
        #             'menengah_threshold': None,
        #             'tidak_sejahtera_threshold': None,
        #             'labeling_method': 'iqr',
        #             'is_reverse': indicator in reverse_indicators
        #         }
                
        #     threshold = LabelingThreshold(**threshold_data)
        #     db.session.add(threshold)
            
        # db.session.commit() 