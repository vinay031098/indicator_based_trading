# ─── Docker image for Railway / Koyeb / Fly.io (all have free tiers) ───
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port (Railway/Koyeb use $PORT env var)
EXPOSE 5000

# Start with gunicorn
CMD gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120
