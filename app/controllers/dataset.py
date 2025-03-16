from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models.indicators import INDICATOR_MODELS
from app import db
from app.services.data_processor import preprocess_indicator_value, label_indicator_value
from app.services.model_trainer import retrain_model_if_needed, generate_predictions
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
    province_filter = request.args.get('province', '')
    year_filter = request.args.get('year', '')
    
    if selected_indicator:
        # Get the model class for the selected indicator
        model_class = INDICATOR_MODELS[selected_indicator]
        
        # Build the query
        query = model_class.query
        
        # Apply filters if provided
        if province_filter:
            query = query.filter(model_class.provinsi.ilike(f'%{province_filter}%'))
        if year_filter:
            query = query.filter(model_class.year == year_filter)
        
        # Get paginated data
        data = query.order_by(model_class.provinsi, model_class.year).paginate(page=page, per_page=per_page)
        
        # Get unique provinces and years for filter dropdowns
        provinces = db.session.query(model_class.provinsi).distinct().order_by(model_class.provinsi).all()
        years = db.session.query(model_class.year).distinct().order_by(model_class.year).all()
    else:
        data = None
        provinces = []
        years = []
    
    return render_template('dataset/index.html', 
                          indicators=indicators,
                          selected_indicator=selected_indicator,
                          data=data,
                          provinces=[p[0] for p in provinces],
                          years=[y[0] for y in years],
                          province_filter=province_filter,
                          year_filter=year_filter)

@dataset_bp.route('/dataset/add', methods=['GET', 'POST'])
@login_required
def add():
    # Only admin can add data
    if not current_user.is_admin:
        flash('You do not have permission to add data', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Get all available indicators
    indicators = list(INDICATOR_MODELS.keys())
    indicators.sort()
    
    if request.method == 'POST':
        indicator = request.form.get('indicator')
        provinsi = request.form.get('provinsi')
        year = request.form.get('year', type=int)
        value = request.form.get('value', type=float)
        
        if not all([indicator, provinsi, year, value is not None]):
            flash('All fields are required', 'danger')
            return render_template('dataset/add.html', indicators=indicators)
        
        # Get the model class for the selected indicator
        model_class = INDICATOR_MODELS[indicator]
        
        # Check if data already exists
        existing_data = model_class.query.filter_by(provinsi=provinsi, year=year).first()
        if existing_data:
            flash(f'Data for {provinsi} in {year} already exists', 'danger')
            return render_template('dataset/add.html', indicators=indicators)
        
        # Preprocess the value
        processed_value = preprocess_indicator_value(indicator, value)
        
        # Label the value
        label = label_indicator_value(indicator, processed_value)
        
        # Create new data
        new_data = model_class(provinsi=provinsi, year=year, value=processed_value, label_sejahtera=label)
        db.session.add(new_data)
        db.session.commit()
        
        # Retrain model if needed
        retrain_model_if_needed(indicator)
        
        flash(f'Data for {provinsi} in {year} added successfully', 'success')
        return redirect(url_for('dataset.index', indicator=indicator))
    
    return render_template('dataset/add.html', indicators=indicators)

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
        retrain_model_if_needed(indicator)
        
        flash(f'Data for {data.provinsi} in {data.year} updated successfully', 'success')
        return redirect(url_for('dataset.index', indicator=indicator))
    
    return render_template('dataset/edit.html', data=data, indicator=indicator)

@dataset_bp.route('/dataset/delete/<indicator>/<int:id>', methods=['POST'])
@login_required
def delete(indicator, id):
    # Only admin can delete data
    if not current_user.is_admin:
        flash('You do not have permission to delete data', 'danger')
        return redirect(url_for('dataset.index'))
    
    # Get the model class for the selected indicator
    model_class = INDICATOR_MODELS[indicator]
    
    # Get the data to delete
    data = model_class.query.get_or_404(id)
    
    # Delete data
    db.session.delete(data)
    db.session.commit()
    
    # Retrain model if needed
    retrain_model_if_needed(indicator)
    
    flash(f'Data for {data.provinsi} in {data.year} deleted successfully', 'success')
    return redirect(url_for('dataset.index', indicator=indicator))

@dataset_bp.route('/dataset/provinces')
@login_required
def get_provinces():
    """API endpoint to get all unique provinces"""
    # Get a sample indicator model
    sample_indicator = list(INDICATOR_MODELS.keys())[0]
    model_class = INDICATOR_MODELS[sample_indicator]
    
    # Get unique provinces
    provinces = db.session.query(model_class.provinsi).distinct().order_by(model_class.provinsi).all()
    
    return jsonify([p[0] for p in provinces])

@dataset_bp.route('/dataset/train-models', methods=['GET', 'POST'])
@login_required
def train_models():
    """Manually trigger model training"""
    # Only admin can trigger model training
    if not current_user.is_admin:
        flash('You do not have permission to train models', 'danger')
        return redirect(url_for('dataset.index'))
    
    if request.method == 'POST':
        from app.services.model_trainer import retrain_model_if_needed, generate_predictions
        success = retrain_model_if_needed('indeks_pembangunan_manusia')
        
        if success:
            # Get the latest model to generate predictions
            from app.models.ml_models import TrainedModel
            latest_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
            
            if latest_model:
                # Generate predictions using the latest model
                predictions = generate_predictions(latest_model.id)
                flash(f'Models trained successfully and {len(predictions)} predictions generated', 'success')
            else:
                flash('Models trained successfully but no model found for predictions', 'warning')
        else:
            flash('Failed to train models', 'danger')
        
        return redirect(url_for('visualization.model_performance'))
    
    return render_template('dataset/train.html') 