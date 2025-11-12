FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code - src folder will be at /app/src/
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/data /tmp/logs

# Set Python path
ENV PYTHONPATH=/app

# Run main.py from src folder
CMD python src/main.py