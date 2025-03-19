# Prosperity Web Project Spec

## Overview

A Flask web application for visualizing and analyzing prosperity indicators across different regions, allowing users to view, add, and edit data, as well as see model performance and visualizations.

## Tech Stack

### Backend

1. **Flask (v2.3+)**

   - Core web framework for the application
   - Blueprints for modular application structure
   - Jinja2 templating for dynamic HTML generation
   - WTForms for form handling and validation
   - Flask-migrate for database migrations

2. **PostgreSQL (v14+)**

   - Relational database for storing all application data
   - Contains tables for users, regions, indicators, and model information
   - Provides transaction support for data integrity
   - Used for storing both raw data and processed analytics results

3. **SQLAlchemy (v2.0+)**

   - ORM for database interactions
   - Handles database connections and connection pooling
   - Provides abstraction layer for database operations
   - Used for defining models with relationships and constraints

4. **Alembic (via Flask-Migrate)**

   - Database migration tool
   - Tracks and applies database schema changes
   - Responsible for initial data loading from preprocessing results
   - Ensures database schema remains in sync with model definitions

5. **Flask-Login**

   - User session management
   - Authentication and authorization handling
   - User role support (admin vs. regular user)
   - Secure password hashing and verification

6. **Scikit-learn (v1.2+)**

   - Machine learning library for model implementation
   - Random Forest and Logistic Regression classifiers
   - Cross-validation and model evaluation tools
   - Feature importance analysis
   - Model serialization via joblib

7. **Pandas (v2.0+)**

   - Data manipulation and preprocessing
   - Handles CSV import/export
   - Provides DataFrame operations for data transformation
   - Used for feature engineering and data cleaning

8. **Matplotlib/Seaborn**

   - Data visualization generation
   - Static chart creation for reports
   - Visualization preprocessing for web display
   - Renders complex statistical visualizations

9. **Flask-WTF**

   - Form validation and CSRF protection
   - Integration with WTForms for form handling
   - Custom field validators for data input

10. **Joblib**

    - Model serialization and persistence
    - Efficient storage of trained ML models
    - Handles loading/saving of model objects

11. **Python-dotenv**
    - Environment variable management
    - Configuration separation for development/production
    - Secure storage of sensitive credentials

### Frontend

1. **HTML5/CSS3/JavaScript (ES6+)**

   - Core web technologies for UI rendering
   - Responsive design principles
   - Cross-browser compatibility

2. **Bootstrap (v5.3+)**

   - Frontend CSS framework
   - Responsive grid system
   - UI components (forms, tables, cards, navigation)
   - Utility classes for quick styling

3. **Chart.js (v4.0+)**

   - Interactive chart library
   - Renders visualizations in browser
   - Supports various chart types (bar, line, radar, pie)
   - Animation and interactivity features
   - Responsive design for different screen sizes

4. **AJAX/Fetch API**

   - Asynchronous data loading
   - Dynamic UI updates without page refresh
   - Used for form submissions and data filtering

5. **DataTables.js**

   - Enhanced HTML tables
   - Client-side pagination
   - Sorting and filtering capabilities
   - Export functionality (CSV, Excel)

6. **Font Awesome**
   - Icon library for UI elements
   - Visual indicators and action buttons

### Development Tools

1. **Docker & Docker Compose**

   - Containerization for consistent environments
   - Multi-container application setup
   - Environment isolation and reproducibility
   - Simplifies deployment and scaling

2. **uv**

   - Python package installation and management
   - Faster alternative to pip
   - Dependency resolution

3. **Git**

   - Version control system
   - Feature branch workflow
   - Code review process

4. **Flask Debug Toolbar**

   - Development debugging assistance
   - Performance profiling
   - Request inspection

5. **pytest**
   - Testing framework for unit and integration tests
   - Fixtures for test setup
   - Coverage reporting

### Data Processing Pipeline

1. **Preprocessing Module**

   - Data cleaning and normalization
   - Feature engineering
   - Data transformation

2. **Model Training Pipeline**

   - Cross-validation
   - Hyperparameter tuning
   - Model evaluation
   - Model selection

3. **Inference Pipeline**
   - Real-time classification
   - Confidence scoring
   - Results storage

## Database Design

### Tables

1. **users**

   - id (PK)
   - username
   - password_hash
   - is_admin

2. **regions**

   - id (PK)
   - name
   - year

3. **indicators** (metadata about indicators)

   - id (PK)
   - name
   - description
   - unit

4. **indicator_values**

   - id (PK)
   - region_id (FK)
   - indicator_id (FK)
   - value
   - year

5. **models**

   - id (PK)
   - model_type (Random Forest/Logistic Regression)
   - trained_date
   - accuracy
   - precision
   - recall
   - f1_score
   - training_time
   - is_best_model

6. **predictions**
   - id (PK)
   - region_id (FK)
   - model_id (FK)
   - classification
   - confidence_score

### Migrations

- Initial migration will create schema and load preprocessed data from the notebook results
- Migration scripts will handle schema changes and data updates

## Web App Structure

### 1. Authentication

#### Routes

1. **/login**

   - [GET] Returns login template
   - [POST] Verifies credentials and redirects to dashboard

2. **/logout**
   - Logs out user and redirects to login

### 2. Dashboard

#### Routes

1. **/** (Dashboard Home)
   - Displays summary cards showing classification results
   - Shows high-level metrics
   - Provides navigation to detailed views

### 3. Dataset Management

#### Routes

1. **/dataset**

   - [GET] Shows tabular data view with:
     - Dropdown for indicator selection
     - Pagination
     - Filtering by region, year
     - Admin users see edit buttons

2. **/dataset/add**

   - [GET] Form for adding new data (admin only)
   - [POST] Validates and stores new data

3. **/dataset/edit/<id>**

   - [GET] Form for editing existing data (admin only)
   - [POST] Updates data

4. **/dataset/delete/<id>**
   - [POST] Removes data entry (admin only)

### 4. Visualizations

#### Routes

1. **/visualization/data**

   - Interactive charts showing indicator distributions
   - Region comparison visualizations
   - Time series visualizations
   - Cards showing classification breakdown

2. **/visualization/model**
   - Model performance comparison dashboard
   - Dropdown to select model type
   - Displays metrics (accuracy, precision, recall, F1)
   - Shows confusion matrix
   - Training details chart

### 5. Model Management

#### Routes

1. **/model/train**

   - [GET] Form to configure model training (admin only)
   - [POST] Triggers model training with selected parameters

2. **/model/predict**
   - Interface for using model to predict for new regions

## User Roles & Permissions

### Admin

- View all data and visualizations
- Add/edit/delete data
- Train models
- Make predictions

### Regular User

- View all data and visualizations
- Make predictions
- Cannot modify data

## Implementation Details

### Data Processing

- Preprocessing functions from the notebook will be adapted into service modules
- Initial data loaded into database during migration

### Model Training

- Models will be trained on demand by admins
- Trained models will be serialized and stored
- Best performing model will be marked

### File Structure

```
prosperity-web/
├── app/
│   ├── __init__.py
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── forms.py
│   │   ├── models.py
│   │   └── routes.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── forms.py
│   │   ├── models.py
│   │   └── routes.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ml_models.py
│   │   ├── forms.py
│   │   ├── models.py
│   │   └── routes.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── routes.py
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   ├── templates/
│   │   ├── auth/
│   │   ├── dataset/
│   │   ├── models/
│   │   ├── visualization/
│   │   ├── base.html
│   │   └── index.html
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   └── visualization.py
│   └── config.py
├── migrations/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── run.py
```

## Docker Configuration

- Multi-container setup with Docker Compose
- Container 1: Flask application
- Container 2: PostgreSQL database
- Dependencies defined in docker-compose.yml
- Automatic database migration on startup

## Deployment Instructions

1. Clone repository
2. Build with `docker-compose build`
3. Start with `docker-compose up`
4. Access application at http://localhost:5000

## Development Notes

1. Use uv for dependency management
