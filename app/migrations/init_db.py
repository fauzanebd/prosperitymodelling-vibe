from flask import Flask
from flask_migrate import Migrate
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import create_app, db
from app.models.user import User
from app.models.indicators import *
from app.models.ml_models import TrainedModel
from app.models.predictions import RegionPrediction

def init_db():
    """Initialize the database schema"""
    print("Initializing database schema...")
    
    # Create app context
    app = create_app()
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create initial users if they don't exist
        if User.query.count() == 0:
            # Create admin user
            admin = User(username='admin', is_admin=True)
            admin.set_password('admin123')
            
            # Create regular user
            user = User(username='user', is_admin=False)
            user.set_password('user123')
            
            db.session.add_all([admin, user])
            db.session.commit()
            print("Initial users created.")
        
        print("Database schema initialized successfully!")

if __name__ == "__main__":
    init_db() 