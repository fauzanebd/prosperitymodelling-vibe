@echo off
setlocal enabledelayedexpansion

REM Load development environment variables
for /F "tokens=*" %%a in ('type .env.dev ^| findstr /V "^#" ^| findstr /V "^$"') do (
    set "%%a"
)

REM Check if PostgreSQL is running in Docker
docker-compose -f docker-compose.dev.yml ps | findstr "db.*Up" > nul
if %ERRORLEVEL% NEQ 0 (
    echo Starting PostgreSQL in Docker...
    docker-compose -f docker-compose.dev.yml up -d
    
    REM Wait for PostgreSQL to be ready
    echo Waiting for PostgreSQL to be ready...
    timeout /t 5 /nobreak > nul
)

REM Initialize the database
python -m app.migrations.init_db

REM Import data
python -m app.migrations.import_data

REM Run the Flask development server on port 5001
set FLASK_RUN_PORT=5001
python run.py --port 5001 