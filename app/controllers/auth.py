from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User
from app import db
from werkzeug.security import generate_password_hash

auth_bp = Blueprint('auth', __name__)

# Create a default user for automatic login
def create_default_user():
    # Create regular user if not exists
    if not User.query.filter_by(username='pengunjung', is_admin=False).first():
        user = User(username='pengunjung', is_admin=False)
        user.set_password('pengunjung123')
        db.session.add(user)
        db.session.commit()
    
    # Create admin if not exists
    if not User.query.filter_by(username='admin', is_admin=True).first():
        admin = User(username='admin', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, just redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    # Auto-login as regular user
    user = User.query.filter_by(username='pengunjung', is_admin=False).first()
    if user:
        login_user(user)
    
    return redirect(url_for('dashboard.index'))

@auth_bp.route('/switch-to-admin', methods=['GET', 'POST'])
@login_required
def switch_to_admin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin_user = User.query.filter_by(username=username, is_admin=True).first()
        
        if admin_user and admin_user.check_password(password):
            logout_user()
            login_user(admin_user)
            flash('Beralih ke akun admin.', 'success')
            return redirect(url_for('dashboard.index'))
        else:
            flash('Invalid admin credentials.', 'danger')
            
    return render_template('auth/admin_login.html')

@auth_bp.route('/switch-to-user')
@login_required
def switch_to_user():
    user = User.query.filter_by(username='pengunjung', is_admin=False).first()
    if user:
        logout_user()
        login_user(user)
        flash('Beralih ke akun pengguna.', 'success')
    else:
        flash('Akun pengguna tidak ditemukan.', 'error')
    return redirect(url_for('dashboard.index'))

@auth_bp.route('/session-check')
def session_check():
    """Debug endpoint to check session state"""
    is_auth = current_user.is_authenticated
    username = current_user.username if is_auth else None
    return {
        'authenticated': is_auth,
        'username': username,
        'endpoint': request.endpoint
    }

# Register the create functions to be called when the app starts
@auth_bp.before_app_request
def init_user_accounts():
    create_default_user()
    
    # Auto-login for non-authenticated users
    if not current_user.is_authenticated and request.endpoint != 'static':
        default_user = User.query.filter_by(username='pengunjung', is_admin=False).first()
        if default_user:
            login_user(default_user) 