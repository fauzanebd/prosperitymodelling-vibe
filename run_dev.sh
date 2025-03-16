#!/bin/bash

# Load development environment variables
export $(cat .env.dev | grep -v '^#' | xargs)

# Start PostgreSQL in Docker if not already running
if ! docker-compose -f docker-compose.dev.yml ps | grep -q "db.*Up"; then
    echo "Starting PostgreSQL in Docker..."
    docker-compose -f docker-compose.dev.yml up -d
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    sleep 5
fi

# Initialize the database
python -m app.migrations.init_db

# Import data
python -m app.migrations.import_data

# Run the Flask development server on port 5001
export FLASK_RUN_PORT=5001
python run.py --port 5001 