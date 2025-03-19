FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=run.py \
    FLASK_ENV=production

# Expose the port
EXPOSE 5000

# Run the application with gunicorn
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "run:app"] 