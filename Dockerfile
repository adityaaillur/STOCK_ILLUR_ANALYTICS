# Base image
FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create and set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Gunicorn config
ENV GUNICORN_CMD_ARGS="--workers=3 --worker-class=uvicorn.workers.UvicornWorker --timeout 300"

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "modules.main:app", "--host", "0.0.0.0", "--port", "8000"]
