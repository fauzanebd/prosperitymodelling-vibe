from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User
from app import db
from werkzeug.security import generate_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard.index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

# Create initial users if they don't exist
@auth_bp.before_app_first_request
def create_initial_users():
    # Check if users already exist
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