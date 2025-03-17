from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models.indicators import INDICATOR_MODELS
from app.models.thresholds import LabelingThreshold
from app import db
from app.services.data_processor import preprocess_indicator_value, label_indicator_value
from app.services.model_trainer import retrain_model_if_needed, generate_predictions, delete_old_models
from sqlalchemy import func
import pandas as pd

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
                          threshold=threshold)

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
            return render_template('dataset/add_for_inference.html', indicators=indicators)
        
        # Check if any data already exists for this region and year
        for indicator in indicators:
            model_class = INDICATOR_MODELS[indicator]
            existing_data = model_class.query.filter_by(region=region, year=year).first()
            if existing_data:
                flash(f'Data for {region} in {year} already exists for {indicator}', 'danger')
                return render_template('dataset/add_for_inference.html', indicators=indicators)
        
        # Process each indicator
        for indicator in indicators:
            value = request.form.get(f'value_{indicator}', type=float)
            
            if value is None:
                flash(f'Value for {indicator} is required', 'danger')
                return render_template('dataset/add_for_inference.html', indicators=indicators)
            
            # Get the model class for the indicator
            model_class = INDICATOR_MODELS[indicator]
            
            # Preprocess the value
            processed_value = preprocess_indicator_value(indicator, value)
            
            # Label the value
            label = label_indicator_value(indicator, processed_value)
            
            # Create new data
            new_data = model_class(region=region, year=year, value=processed_value, label_sejahtera=label)
            db.session.add(new_data)
        
        # Commit all changes
        db.session.commit()
        
        # Generate predictions for the new region
        from app.models.ml_models import TrainedModel
        from app.models.predictions import RegionPrediction
        best_model = TrainedModel.query.order_by(TrainedModel.accuracy.desc()).first()
        
        if best_model:
            # Delete any existing predictions for this region and year
            RegionPrediction.query.filter_by(region=region, year=year).delete()
            db.session.commit()
            
            # Generate predictions using the best model
            predictions = generate_predictions(best_model.id)
            flash(f'Data for {region} in {year} added successfully and predictions updated', 'success')
        else:
            flash(f'Data for {region} in {year} added successfully, but no model found for predictions', 'warning')
        
        return redirect(url_for('dashboard.index'))
    
    return render_template('dataset/add_for_inference.html', indicators=indicators)

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
            return render_template('dataset/add_for_training.html', indicators=indicators)
        
        # Process each year and indicator
        years = range(2019, 2024)
        for year in years:
            for indicator in indicators:
                value = request.form.get(f'value_{year}_{indicator}', type=float)
                
                if value is None:
                    flash(f'Value for {indicator} in {year} is required', 'danger')
                    return render_template('dataset/add_for_training.html', indicators=indicators)
                
                # Get the model class for the indicator
                model_class = INDICATOR_MODELS[indicator]
                
                # Check if data already exists
                existing_data = model_class.query.filter_by(region=region, year=year).first()
                if existing_data:
                    # Update existing data
                    processed_value = preprocess_indicator_value(indicator, value)
                    label = label_indicator_value(indicator, processed_value)
                    
                    existing_data.value = processed_value
                    existing_data.label_sejahtera = label
                else:
                    # Create new data
                    processed_value = preprocess_indicator_value(indicator, value)
                    label = label_indicator_value(indicator, processed_value)
                    
                    new_data = model_class(region=region, year=year, value=processed_value, label_sejahtera=label)
                    db.session.add(new_data)
        
        # Commit all changes
        db.session.commit()
        
        # Retrain model
        retrain_model_if_needed('indeks_pembangunan_manusia')
        
        flash(f'Data for {region} for all years added successfully, model retrained, and old models cleaned up', 'success')
        return redirect(url_for('visualization.model_performance'))
    
    return render_template('dataset/add_for_training.html', indicators=indicators)

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
            return render_template('dataset/edit.html', data=data, indicator=indicator)
        
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
    
    return render_template('dataset/edit.html', data=data, indicator=indicator)

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