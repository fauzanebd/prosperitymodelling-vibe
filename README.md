# Sejahteraku Web Application

This web application provides a user interface for analyzing regional prosperity based on various socioeconomic indicators. It allows users to view, add, and edit indicator data, as well as visualize the data and model performance.

## Features

- User authentication with admin and regular user roles
- Dataset management (view, add, edit, delete)
- Data visualization dashboard
- Model performance comparison
- Prosperity prediction for regions

## Requirements

- Python 3.9+
- PostgreSQL
- Docker and Docker Compose (optional)
- uv (Python package installer)

## Installation

### Using Docker (Recommended)

1. Clone the repository:

   ```
   git clone <repository-url>
   cd kesejahteraan
   ```

2. Build and run the application using Docker Compose:

   ```
   docker-compose up --build
   ```

3. Access the application at http://localhost:5000

### Manual Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd kesejahteraan
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install uv if you don't have it:

   ```
   pip install uv
   ```

4. Install the dependencies using uv:

   ```
   uv pip install -e .
   ```

5. Set up the PostgreSQL database and update the `DATABASE_URL` in the `.env` file.

6. Initialize the database:

   ```
   python -m app.migrations.init_db
   ```

7. Import the data:

   ```
   python -m app.migrations.import_data
   ```

8. Run the application:

   ```
   python run.py
   ```

   Or use the provided script:

   ```
   ./run_app.sh
   ```

9. Access the application at http://localhost:5000

## Usage

### Authentication

- Admin user: username `admin`, password `admin123`
- Regular user: username `user`, password `user123`

### Manajemen Data

- View all indicators data with filtering and pagination
- Add new data for indicators (admin only)
- Edit existing data (admin only)
- Delete data (admin only)

### Visualization

- Model performance comparison between Random Forest and Logistic Regression
- Data visualization for each indicator
- Prosperity prediction visualization

## Project Structure

- `app/`: Main application package
  - `controllers/`: Route handlers
  - `models/`: Database models
  - `services/`: Business logic
  - `templates/`: HTML templates
  - `static/`: Static files (CSS, JS, images)
  - `migrations/`: Database migrations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
