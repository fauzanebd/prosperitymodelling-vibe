import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from app.models.ml_models import TrainedModel
from app.models.predictions import RegionPrediction
from app.services.data_processor import create_combined_dataset, prepare_data_for_model
from app import db

def train_random_forest(X, y, feature_names):
    """
    Train a Random Forest model
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features for model training
    y : pd.Series
        Target variable for model training
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary containing the trained model, metrics, and parameters
    """
    # Start timer
    start_time = time.time()
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=5)
    
    # Train the model on the full dataset
    model.fit(X_scaled, y)
    
    # Calculate metrics
    cm = confusion_matrix(y, y_pred_cv)
    report = classification_report(y, y_pred_cv, output_dict=True)
    
    # End timer
    end_time = time.time()
    training_time = end_time - start_time
    
    # Create results dictionary
    results = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1_score': report['1']['f1-score'] if '1' in report else 0,
            'training_time': training_time,
            'confusion_matrix': cm
        },
        'parameters': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    }
    
    return results

def train_logistic_regression(X, y, feature_names):
    """
    Train a Logistic Regression model
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features for model training
    y : pd.Series
        Target variable for model training
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary containing the trained model, metrics, and parameters
    """
    # Start timer
    start_time = time.time()
    
    # Create and train the model
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=5)
    
    # Train the model on the full dataset
    model.fit(X_scaled, y)
    
    # Calculate metrics
    cm = confusion_matrix(y, y_pred_cv)
    report = classification_report(y, y_pred_cv, output_dict=True)
    
    # End timer
    end_time = time.time()
    training_time = end_time - start_time
    
    # Create results dictionary
    results = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1_score': report['1']['f1-score'] if '1' in report else 0,
            'training_time': training_time,
            'confusion_matrix': cm
        },
        'parameters': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
    }
    
    return results

def save_model_to_db(model_type, results):
    """
    Save a trained model to the database
    
    Parameters:
    -----------
    model_type : str
        Type of model ('random_forest' or 'logistic_regression')
    results : dict
        Dictionary containing the trained model, metrics, and parameters
        
    Returns:
    --------
    TrainedModel
        The saved model
    """
    # Create a new TrainedModel instance
    trained_model = TrainedModel(model_type=model_type)
    
    # Save model and related data
    trained_model.save_model(
        results['model'],
        results['scaler'],
        results['feature_names'],
        results['metrics'],
        results['parameters']
    )
    
    # Save to database
    db.session.add(trained_model)
    db.session.commit()
    
    return trained_model

def generate_predictions(model_id):
    """
    Generate predictions for all regions using a trained model
    
    Parameters:
    -----------
    model_id : int
        ID of the trained model to use
        
    Returns:
    --------
    list
        List of RegionPrediction objects
    """
    # Get the trained model
    trained_model = TrainedModel.query.get(model_id)
    if not trained_model:
        print(f"Model with ID {model_id} not found")
        return []
    
    # Load the model, scaler, and feature names
    model, scaler, feature_names = trained_model.load_model()
    
    # Get the latest year with data
    latest_year = 2023  # Default to 2023
    
    # Create a combined dataset for the latest year
    combined_df = create_combined_dataset(latest_year)
    if combined_df is None:
        print(f"No data found for year {latest_year}")
        return []
    
    # Prepare data for prediction
    X, _, _ = prepare_data_for_model(combined_df, 'indeks_pembangunan_manusia')
    if X is None:
        print("Failed to prepare data for prediction")
        return []
    
    # Ensure X has the same features as the model was trained on
    missing_features = set(feature_names) - set(X.columns)
    for feature in missing_features:
        X[feature] = 0  # Fill missing features with 0
    
    X = X[feature_names]  # Reorder columns to match feature_names
    
    # Standardize features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_scaled)
        y_pred = model.predict(X_scaled)
    else:
        y_pred = model.predict(X_scaled)
        y_prob = np.zeros((len(y_pred), 2))
        y_prob[np.arange(len(y_pred)), y_pred] = 1
    
    # Map predictions to classes
    class_mapping = {1: 'Sejahtera', 0: 'Menengah'}
    predicted_classes = [class_mapping[p] for p in y_pred]
    
    # Create RegionPrediction objects
    predictions = []
    for i, row in combined_df.reset_index().iterrows():
        prediction = RegionPrediction(
            provinsi=row['wilayah'],
            year=latest_year,
            model_id=model_id,
            predicted_class=predicted_classes[i],
            prediction_probability=y_prob[i, 1] if y_pred[i] == 1 else y_prob[i, 0]
        )
        predictions.append(prediction)
    
    # Save predictions to database
    db.session.add_all(predictions)
    db.session.commit()
    
    return predictions

def retrain_model_if_needed(indicator_name):
    """
    Retrain models if data for a specific indicator has changed
    
    Parameters:
    -----------
    indicator_name : str
        Name of the indicator that has changed
        
    Returns:
    --------
    bool
        True if models were retrained, False otherwise
    """
    # Only retrain if the indicator is indeks_pembangunan_manusia or if it's a feature used by the model
    if indicator_name != 'indeks_pembangunan_manusia':
        # Check if there are any trained models
        if TrainedModel.query.count() == 0:
            return False
        
        # Get the latest model
        latest_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
        
        # Load the model, scaler, and feature names
        _, _, feature_names = latest_model.load_model()
        
        # Check if the indicator is used as a feature
        if indicator_name not in feature_names:
            return False
    
    # Get the latest year with data
    latest_year = 2023  # Default to 2023
    
    # Create a combined dataset for the latest year
    combined_df = create_combined_dataset(latest_year)
    if combined_df is None:
        print(f"No data found for year {latest_year}")
        return False
    
    # Prepare data for model training
    X, y, feature_names = prepare_data_for_model(combined_df, 'indeks_pembangunan_manusia')
    if X is None or y is None:
        print("Failed to prepare data for model training")
        return False
    
    # Train Random Forest model
    rf_results = train_random_forest(X, y, feature_names)
    rf_model = save_model_to_db('random_forest', rf_results)
    
    # Train Logistic Regression model
    lr_results = train_logistic_regression(X, y, feature_names)
    lr_model = save_model_to_db('logistic_regression', lr_results)
    
    # Generate predictions using the Random Forest model
    generate_predictions(rf_model.id)
    
    return True 