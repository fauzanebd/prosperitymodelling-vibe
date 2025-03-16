#!/bin/bash

# Initialize the database
python -m app.migrations.init_db

# Import data
python -m app.migrations.import_data

# Run the application
python run.py 