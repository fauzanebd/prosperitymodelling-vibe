from app import db
import pickle
import json
from datetime import datetime
import time
import numpy as np

class TrainedModel(db.Model):
    __tablename__ = 'trained_models'
    
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(64))  # 'random_forest' or 'logistic_regression'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_binary = db.Column(db.LargeBinary)  # Pickled model
    scaler_binary = db.Column(db.LargeBinary)  # Pickled scaler
    feature_names = db.Column(db.Text)  # JSON string of feature names
    
    # Performance metrics
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)  # Added test accuracy
    training_time = db.Column(db.Float)  # in seconds
    inference_time = db.Column(db.Float)  # in seconds
    confusion_matrix = db.Column(db.Text)  # JSON string of confusion matrix
    feature_importance = db.Column(db.Text)  # JSON string of feature importance (for RF)
    cv_scores = db.Column(db.Text)  # JSON string of cross-validation scores
    mean_cv_accuracy = db.Column(db.Float)  # Mean of cross-validation scores
    std_cv_accuracy = db.Column(db.Float)  # Standard deviation of cross-validation scores
    
    # Training parameters
    training_parameters = db.Column(db.Text)  # JSON string of training parameters
    
    def save_model(self, model, scaler, feature_names, metrics, parameters):
        """Save model and related data to database"""
        self.model_binary = pickle.dumps(model)
        self.scaler_binary = pickle.dumps(scaler)
        self.feature_names = json.dumps(feature_names)
        
        # Save metrics
        self.accuracy = float(metrics.get('accuracy'))
        self.precision = float(metrics.get('precision'))
        self.recall = float(metrics.get('recall'))
        self.f1_score = float(metrics.get('f1_score'))
        self.test_accuracy = float(metrics.get('test_accuracy'))  # Added test accuracy
        self.training_time = float(metrics.get('training_time'))
        self.confusion_matrix = json.dumps(metrics.get('confusion_matrix', []).tolist())
        
        # Save cross-validation scores
        self.cv_scores = json.dumps(metrics.get('cv_scores', []))
        self.mean_cv_accuracy = float(metrics.get('mean_cv_accuracy'))
        self.std_cv_accuracy = float(metrics.get('std_cv_accuracy'))
        
        # Calculate inference time
        if model is not None and scaler is not None:
            # Create a small sample for inference time calculation
            sample_size = 100
            n_features = len(feature_names)
            X_sample = np.random.rand(sample_size, n_features)
            X_scaled = scaler.transform(X_sample)
            
            # Measure inference time
            start_time = time.time()
            model.predict(X_scaled)
            end_time = time.time()
            
            # Calculate average inference time per sample
            self.inference_time = (end_time - start_time) / sample_size
        else:
            self.inference_time = 0.0
        
        # Save feature importance
        if 'feature_importance' in metrics and metrics['feature_importance']:
            self.feature_importance = json.dumps(metrics['feature_importance'])
        elif self.model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            self.feature_importance = json.dumps(feature_importance)
        
        # Save training parameters
        self.training_parameters = json.dumps(parameters)
    
    def load_model(self):
        """Load model and scaler from database"""
        model = pickle.loads(self.model_binary)
        scaler = pickle.loads(self.scaler_binary)
        feature_names = json.loads(self.feature_names)
        return model, scaler, feature_names
    
    def get_metrics(self):
        """Get model metrics as a dictionary"""
        metrics = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'test_accuracy': self.test_accuracy,  # Added test accuracy
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'confusion_matrix': json.loads(self.confusion_matrix) if self.confusion_matrix else None,
            'feature_importance': json.loads(self.feature_importance) if self.feature_importance else None,
            'cv_scores': json.loads(self.cv_scores) if self.cv_scores else [],
            'mean_cv_accuracy': self.mean_cv_accuracy,
            'std_cv_accuracy': self.std_cv_accuracy
        }
        return metrics
    
    def get_parameters(self):
        """Get training parameters as a dictionary"""
        return json.loads(self.training_parameters) if self.training_parameters else {} 