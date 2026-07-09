"""Vercel serverless entry point — exposes the FastAPI app as an ASGI handler."""

import os
import sys

# Ensure the project root is on sys.path so absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Vercel sets this via environment; provide a fallback so imports don't crash
os.environ.setdefault("NVIDIA_API_KEY", "")

from app.main import app

handler = app
