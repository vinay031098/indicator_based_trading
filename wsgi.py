"""WSGI entry point for Gunicorn (production)."""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from app import app, auto_connect
import threading

# Auto-connect on startup
threading.Timer(2, auto_connect).start()

if __name__ == "__main__":
    app.run()
