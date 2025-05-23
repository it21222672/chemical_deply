FROM python:3.10.0-slim

# 1. Install critical system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up working directory
WORKDIR /app

# 3. Copy requirements file
COPY requirements.txt .

# 4. Install Python dependencies (Flask was missing before!)
# Replace the existing install line
RUN pip install --no-cache-dir -r requirements.txt


# 5. Copy application code
COPY ./app /app

# 6. Expose port and run the app
EXPOSE 5000
CMD ["python", "ml.py"]
