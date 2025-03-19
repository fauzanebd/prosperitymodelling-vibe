from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models.indicators import INDICATOR_MODELS
from app.models.thresholds import LabelingThreshold
from app import db
from app.services.data_processor import preprocess_indicator_value, label_indicator_value_for_inference, label_indicator_value_for_training
from app.services.model_trainer import retrain_model_if_needed, generate_predictions, delete_old_models
from sqlalchemy import func
import pandas as pd
import numpy as np

dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route('/dataset')
@login_required
def index():
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    # Get the selected indicator (default to the first one)
    selected_indicator = request.args.get('indicator', indicators[0] if indicators else None)
    
    # Get page number for pagination
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Get filter parameters
    region_filter = request.args.get('region', '')
    year_filter = request.args.get('year', '')
    
    # Get labeling threshold for the selected indicator
    threshold = None
    if selected_indicator:
        threshold = LabelingThreshold.query.filter_by(indicator=selected_indicator).first()
    
    if selected_indicator:
        # Get the model class for the selected indicator
        model_class = INDICATOR_MODELS[selected_indicator]
        
        # Build the query
        query = model_class.query
        
        # Apply filters if provided
        if region_filter:
            query = query.filter(model_class.region.ilike(f'%{region_filter}%'))
        if year_filter:
            query = query.filter(model_class.year == year_filter)
        
        # Get paginated data
        data = query.order_by(model_class.region, model_class.year).paginate(page=page, per_page=per_page)
        
        # Get unique regions and years for filter dropdowns
        regions = db.session.query(model_class.region).distinct().order_by(model_class.region).all()
        years = db.session.query(model_class.year).distinct().order_by(model_class.year).all()
    else:
        data = None
        regions = []
        years = []
    
    return render_template('dataset/index.html', 
                          indicators=indicators,
                          selected_indicator=selected_indicator,
                          data=data,
                          regions=[p[0] for p in regions],
                          years=[y[0] for y in years],
                          region_filter=region_filter,
                          year_filter=year_filter,
                          threshold=threshold,
                          INDICATOR_MODELS=INDICATOR_MODELS)

@dataset_bp.route('/dataset/add', methods=['GET', 'POST'])
@login_required
def add():
    """This route is deprecated. Redirects to dataset index."""
    flash('Single data point insertion is no longer supported. Please use "Add Data for Inference" or "Add Data for Training" options.', 'warning')
    return redirect(url_for('dataset.index'))

@dataset_bp.route('/dataset/add-for-inference', methods=['GET', 'POST'])
@login_required
def add_for_inference():
    """Add data for a new region for inference only"""
    # Only admin can add data
    if not current_user.is_admin:
        flash('You do not have permission to add data', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Get all available indicators except indeks_pembangunan_manusia
    indicators = list(INDICATOR_MODELS.keys())
    if 'indeks_pembangunan_manusia' in indicators:
        indicators.remove('indeks_pembangunan_manusia')
    indicators.sort()
    
    if request.method == 'POST':
        region = request.form.get('region')
        year = request.form.get('year', type=int)
        
        if not all([region, year]):
            flash('Region and year are required', 'danger')
            return render_template('dataset/add_for_inference.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)
        
        # Check if any data already exists for this region and year
        for indicator in indicators:
            model_class = INDICATOR_MODELS[indicator]
            existing_data = model_class.query.filter_by(region=region, year=year).first()
            if existing_data:
                flash(f'Data for {region} in {year} already exists for {indicator}', 'danger')
                return render_template('dataset/add_for_inference.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)
        
        # Process each indicator
        for indicator in indicators:
            value = request.form.get(f'value_{indicator}', type=float)
            
            if value is None:
                flash(f'Value for {indicator} is required', 'danger')
                return render_template('dataset/add_for_inference.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)
            
            # Get the model class for the indicator
            model_class = INDICATOR_MODELS[indicator]
            
            # Preprocess the value
            processed_value = preprocess_indicator_value(indicator, value)
            
            # Label the value
            label = label_indicator_value_for_inference(indicator, processed_value)
            
            # Create new data
            new_data = model_class(region=region, year=year, value=processed_value, label_sejahtera=label)
            db.session.add(new_data)
        
        # Commit all changes
        db.session.commit()
        
        # Get the best model for prediction
        from app.models.ml_models import TrainedModel
        from app.models.predictions import RegionPrediction
        best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
        
        if best_model:
            # Delete any existing predictions for this region and year
            RegionPrediction.query.filter_by(region=region, year=year).delete()
            db.session.commit()
            
            # Generate prediction just for this region and year
            # Create a combined dataset for this region
            from app.services.data_processor import create_combined_dataset_for_region
            region_df = create_combined_dataset_for_region(region, year)
            
            if region_df is not None:
                # Load the model, scaler, and feature names
                model, scaler, feature_names = best_model.load_model()
                
                # Prepare the data for prediction
                from app.services.data_processor import prepare_data_for_model
                X, _, _ = prepare_data_for_model(region_df, 'indeks_pembangunan_manusia')
                
                if X is not None:
                    # Add year as a feature
                    X['year'] = year
                    
                    # Ensure X has the same features as the model was trained on
                    missing_features = set(feature_names) - set(X.columns)
                    for feature in missing_features:
                        X[feature] = 0  # Fill missing features with 0
                    
                    extra_features = set(X.columns) - set(feature_names)
                    if extra_features:
                        X = X.drop(columns=extra_features)
                    
                    X = X[feature_names]  # Reorder columns to match feature_names
                    
                    # Standardize features
                    X_scaled = scaler.transform(X)
                    
                    # Make prediction
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_scaled)
                        y_pred = model.predict(X_scaled)
                        
                        # Check if y_prob has enough columns for all classes
                        if y_prob.shape[1] <= int(y_pred.max()):
                            # Not enough columns, create a new array with more columns
                            new_y_prob = np.zeros((len(y_pred), int(y_pred.max()) + 1))
                            # Copy existing probabilities
                            for i in range(min(y_prob.shape[1], new_y_prob.shape[1])):
                                new_y_prob[:, i] = y_prob[:, i]
                            # Set the probability for the predicted class to 1.0
                            for i, pred in enumerate(y_pred):
                                new_y_prob[i, int(pred)] = 1.0
                            y_prob = new_y_prob
                    else:
                        y_pred = model.predict(X_scaled)
                        # Create a zeros array with enough columns for all classes
                        max_class = 2  # For Sejahtera (0, 1, 2)
                        y_prob = np.zeros((len(y_pred), max_class + 1))
                        for i, pred in enumerate(y_pred):
                            y_prob[i, int(pred)] = 1.0
                    
                    # Map prediction to class
                    class_mapping = {2: 'Sejahtera', 1: 'Menengah', 0: 'Tidak Sejahtera'}
                    predicted_class = class_mapping[int(y_pred[0])]
                    
                    # Create RegionPrediction object
                    # Use a more robust approach for getting the probability
                    prediction_prob = y_prob[0, int(y_pred[0])] if int(y_pred[0]) < y_prob.shape[1] else 1.0
                    
                    prediction = RegionPrediction(
                        region=region,
                        year=year,
                        model_id=best_model.id,
                        predicted_class=predicted_class,
                        prediction_probability=float(prediction_prob)
                    )
                    
                    db.session.add(prediction)
                    db.session.commit()
                    
                    # Redirect to a new page to show the prediction result
                    return redirect(url_for('dataset.prediction_result', region=region, year=year))
            
            flash(f'Data for {region} in {year} added successfully, but prediction failed', 'warning')
        else:
            flash(f'Data for {region} in {year} added successfully, but no model found for prediction', 'warning')
        
        return redirect(url_for('dashboard.index'))
    
    return render_template('dataset/add_for_inference.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)

@dataset_bp.route('/dataset/prediction_result/<region>/<int:year>')
@login_required
def prediction_result(region, year):
    """Show prediction result for a specific region and year"""
    # Get the prediction for this region and year
    from app.models.predictions import RegionPrediction
    from app.models.ml_models import TrainedModel
    
    prediction = RegionPrediction.query.filter_by(region=region, year=year).order_by(RegionPrediction.id.desc()).first()
    
    if not prediction:
        flash(f'No prediction found for {region} in {year}', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get the model used for prediction
    model = TrainedModel.query.get(prediction.model_id)
    
    # Get all indicator values for this region and year
    indicator_values = {}
    for indicator, model_class in INDICATOR_MODELS.items():
        data = model_class.query.filter_by(region=region, year=year).first()
        if data:
            indicator_values[indicator] = {
                'value': data.value,
                'label': data.label_sejahtera,
                'unit': model_class.unit
            }
    
    return render_template('dataset/prediction_result.html', 
                          prediction=prediction, 
                          model=model, 
                          region=region, 
                          year=year,
                          indicator_values=indicator_values)

@dataset_bp.route('/dataset/inference_predictions')
@login_required
def inference_predictions():
    """Display a list of predictions made for inference-only data (not used in training)"""
    # Get page number for pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Get filter parameters
    region_filter = request.args.get('region', '')
    year_filter = request.args.get('year', '')
    
    # Get the best model based on accuracy
    from app.models.ml_models import TrainedModel
    from app.models.predictions import RegionPrediction
    best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
    
    if not best_model:
        flash('No trained models available', 'warning')
        return render_template('dataset/inference_predictions.html', 
                              predictions=None,
                              regions=[],
                              years=[],
                              region_filter='',
                              year_filter='')
    
    # Query to identify inference-only regions
    # These are regions that have data for some indicators but not for indeks_pembangunan_manusia
    from app.models.indicators import AngkaHarapanHidup  # Use any indicator model as a base
    
    # Get all unique regions with data
    regions_with_data = db.session.query(AngkaHarapanHidup.region).distinct().all()
    regions_with_data = [r[0] for r in regions_with_data]
    
    # Get regions with indeks_pembangunan_manusia data (training data)
    from app.models.indicators import IndeksPembangunanManusia
    regions_with_ipm = db.session.query(IndeksPembangunanManusia.region).distinct().all()
    regions_with_ipm = [r[0] for r in regions_with_ipm]
    
    # Find inference-only regions (have data but no IPM data)
    inference_only_regions = list(set(regions_with_data) - set(regions_with_ipm))
    
    # Build the query for predictions
    query = RegionPrediction.query.filter(
        RegionPrediction.model_id == best_model.id,
        RegionPrediction.region.in_(inference_only_regions)
    )
    
    # Apply filters if provided
    if region_filter:
        query = query.filter(RegionPrediction.region.ilike(f'%{region_filter}%'))
    if year_filter:
        query = query.filter(RegionPrediction.year == year_filter)
    
    # Get paginated data
    predictions = query.order_by(RegionPrediction.region, RegionPrediction.year).paginate(page=page, per_page=per_page)
    
    # Get unique regions and years for filter dropdowns
    regions = db.session.query(RegionPrediction.region).filter(
        RegionPrediction.region.in_(inference_only_regions)
    ).distinct().order_by(RegionPrediction.region).all()
    
    years = db.session.query(RegionPrediction.year).filter(
        RegionPrediction.region.in_(inference_only_regions)
    ).distinct().order_by(RegionPrediction.year).all()
    
    return render_template('dataset/inference_predictions.html', 
                          predictions=predictions,
                          regions=[r[0] for r in regions],
                          years=[y[0] for y in years],
                          region_filter=region_filter,
                          year_filter=year_filter,
                          model=best_model)

@dataset_bp.route('/dataset/add-for-training', methods=['GET', 'POST'])
@login_required
def add_for_training():
    """Add complete data for a region for model training"""
    # Only admin can add data
    if not current_user.is_admin:
        flash('You do not have permission to add data', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    if request.method == 'POST':
        region = request.form.get('region')
        
        if not region:
            flash('Region is required', 'danger')
            return render_template('dataset/add_for_training.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)
        
        # Process each year and indicator
        years = range(2019, 2024)
        for year in years:
            for indicator in indicators:
                value = request.form.get(f'value_{year}_{indicator}', type=float)
                
                if value is None:
                    flash(f'Value for {indicator} in {year} is required', 'danger')
                    return render_template('dataset/add_for_training.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)
                
                # Get the model class for the indicator
                model_class = INDICATOR_MODELS[indicator]
                
                # Check if data already exists
                existing_data = model_class.query.filter_by(region=region, year=year).first()
                if existing_data:
                    # Update existing data
                    processed_value = preprocess_indicator_value(indicator, value)
                    label = label_indicator_value_for_training(indicator, processed_value)
                    
                    existing_data.value = processed_value
                    existing_data.label_sejahtera = label
                else:
                    # Create new data
                    processed_value = preprocess_indicator_value(indicator, value)
                    label = label_indicator_value_for_training(indicator, processed_value)
                    
                    new_data = model_class(region=region, year=year, value=processed_value, label_sejahtera=label)
                    db.session.add(new_data)
        
        # Commit all changes
        db.session.commit()
        
        # Retrain model
        retrain_result = retrain_model_if_needed('indeks_pembangunan_manusia')
        
        if retrain_result:
            flash(f'Data for {region} for all years added successfully and model retrained', 'success')
            
            # Get the most recent year with data
            most_recent_year = max(years)
            
            # Get the latest model to generate a prediction for the most recent year
            from app.models.ml_models import TrainedModel
            from app.models.predictions import RegionPrediction
            best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
            
            if best_model:
                # Get the prediction for the most recent year
                prediction = RegionPrediction.query.filter_by(
                    region=region, 
                    year=most_recent_year,
                    model_id=best_model.id
                ).first()
                
                if prediction:
                    # Redirect to the prediction result page
                    return redirect(url_for('dataset.prediction_result', region=region, year=most_recent_year))
        else:
            flash(f'Data for {region} for all years added successfully but model training failed', 'warning')
        
        return redirect(url_for('visualization.model_performance'))
    
    return render_template('dataset/add_for_training.html', indicators=indicators, INDICATOR_MODELS=INDICATOR_MODELS)

@dataset_bp.route('/dataset/edit/<indicator>/<int:id>', methods=['GET', 'POST'])
@login_required
def edit(indicator, id):
    # Only admin can edit data
    if not current_user.is_admin:
        flash('You do not have permission to edit data', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Get the model class for the selected indicator
    model_class = INDICATOR_MODELS[indicator]
    
    # Get the data to edit
    data = model_class.query.get_or_404(id)
    
    if request.method == 'POST':
        value = request.form.get('value', type=float)
        
        if value is None:
            flash('Value is required', 'danger')
            return render_template('dataset/edit.html', data=data, indicator=indicator, INDICATOR_MODELS=INDICATOR_MODELS)
        
        # Preprocess the value
        processed_value = preprocess_indicator_value(indicator, value)
        
        # Label the value
        label = label_indicator_value(indicator, processed_value)
        
        # Update data
        data.value = processed_value
        data.label_sejahtera = label
        db.session.commit()
        
        # Retrain model if needed
        retrain_result = retrain_model_if_needed(indicator)
        
        if retrain_result:
            flash(f'Data for {data.region} in {data.year} updated successfully, model retrained, and old models cleaned up', 'success')
        else:
            flash(f'Data for {data.region} in {data.year} updated successfully', 'success')
        
        return redirect(url_for('dataset.index', indicator=indicator))
    
    return render_template('dataset/edit.html', data=data, indicator=indicator, INDICATOR_MODELS=INDICATOR_MODELS)

@dataset_bp.route('/dataset/delete/<indicator>/<int:id>', methods=['POST'])
@login_required
def delete(indicator, id):
    """This route is deprecated. Redirects to dataset index."""
    flash('Single data point deletion is no longer supported. Please use "Delete Region Data" option to delete all data for a region.', 'warning')
    return redirect(url_for('dataset.index'))

@dataset_bp.route('/dataset/delete-region', methods=['POST'])
@login_required
def delete_region():
    """Delete all data for a region across all indicators and years"""
    # Only admin can delete data
    if not current_user.is_admin:
        flash('You do not have permission to delete data', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Get the region from query parameters
    region = request.args.get('region')
    
    if not region:
        flash('Region is required', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Delete data for all indicators and years for the region
    deleted_count = 0
    for indicator, model_class in INDICATOR_MODELS.items():
        # Get all data for the region
        data_to_delete = model_class.query.filter_by(region=region).all()
        
        # Delete each data point
        for data in data_to_delete:
            db.session.delete(data)
            deleted_count += 1
    
    # Commit all changes
    db.session.commit()
    
    # Retrain model
    retrain_result = retrain_model_if_needed('indeks_pembangunan_manusia')
    
    if retrain_result:
        flash(f'All data for {region} deleted successfully ({deleted_count} data points), model retrained, and old models cleaned up', 'success')
    else:
        flash(f'All data for {region} deleted successfully ({deleted_count} data points)', 'success')
    
    return redirect(url_for('dataset.index'))

@dataset_bp.route('/dataset/regions')
@login_required
def get_regions():
    """API endpoint to get all unique regions"""
    # Get a sample indicator model
    sample_indicator = list(INDICATOR_MODELS.keys())[0]
    model_class = INDICATOR_MODELS[sample_indicator]
    
    # Get unique regions
    regions = db.session.query(model_class.region).distinct().order_by(model_class.region).all()
    
    return jsonify([p[0] for p in regions])

@dataset_bp.route('/dataset/train-models', methods=['GET', 'POST'])
@login_required
def train_models():
    """Manually trigger model training"""
    # Only admin can trigger model training
    if not current_user.is_admin:
        flash('You do not have permission to train models', 'danger')
        return redirect(url_for('dataset.index'))
    
    if request.method == 'POST':
        from app.services.model_trainer import retrain_model_if_needed, generate_predictions, delete_old_models
        success = retrain_model_if_needed('indeks_pembangunan_manusia')
        
        if success:
            # Get the latest model to generate predictions
            from app.models.ml_models import TrainedModel
            latest_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
            
            if latest_model:
                # Generate predictions using the latest model
                predictions = generate_predictions(latest_model.id)
                flash(f'Models trained successfully, old models cleaned up, and {len(predictions)} predictions generated', 'success')
            else:
                flash('Models trained successfully but no model found for predictions', 'warning')
        else:
            flash('Failed to train models', 'danger')
        
        return redirect(url_for('visualization.model_performance'))
    
    return render_template('dataset/train.html') 