FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy pyproject.toml
COPY pyproject.toml .

# Install Python dependencies using uv
RUN uv pip install --no-cache-dir -e .

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"] 