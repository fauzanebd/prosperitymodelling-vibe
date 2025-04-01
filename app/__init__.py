from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import os
import re

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

def format_indicator(value):
    """
    Custom filter to format indicator names.
    Replaces underscores with spaces and properly formats acronyms like PDRB.
    """
    # Replace underscores with spaces
    value = value.replace('_', ' ')
    
    # Special case for PDRB acronym
    value = re.sub(r'\bpdrb\b', 'PDRB', value, flags=re.IGNORECASE)
    
    # Title case everything else
    words = value.split()
    result = []
    for word in words:
        if word.lower() != 'pdrb':
            word = word.capitalize()
        result.append(word)
    
    return ' '.join(result)

def create_app(config=None):
    app = Flask(__name__)
    
    # Configure the app
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_change_in_production'),
        SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/prosperity'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    
    # Override config if provided
    if config:
        app.config.from_mapping(config)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'dashboard.index'
    login_manager.login_message = None  # Remove login message for auto-login
    login_manager.session_protection = None  # Disable advanced session protection for auto-login
    
    # Register custom Jinja filters
    app.jinja_env.filters['format_indicator'] = format_indicator
    
    # Register blueprints
    from app.controllers.auth import auth_bp
    from app.controllers.dashboard import dashboard_bp
    from app.controllers.dataset import dataset_bp
    from app.controllers.visualization import visualization_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(visualization_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app 