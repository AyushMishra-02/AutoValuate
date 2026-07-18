FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies required for LightGBM (libgomp1)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn (binds to the PORT environment variable provided by cloud platforms)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
