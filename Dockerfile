# Use stable python 3.12 slim bookworm base image
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage docker caching layers
COPY requirements.txt .

# Install dependencies using system pip (pre-built wheels match easily)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure empty directories exist for serving static assets
RUN mkdir -p app/static/uploads app/static/heatmaps app/static/reports

# Expose port
EXPOSE 5000

# Start production server using Gunicorn
# Using 1 worker for ML serving in low-resource environments (e.g. Render, Railway free tiers)
CMD ["gunicorn", "--workers=1", "--threads=2", "--bind=0.0.0.0:5000", "--timeout=120", "run:app"]
