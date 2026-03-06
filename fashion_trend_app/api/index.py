import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change working directory to app root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_with_progress import app

# Vercel expects the WSGI app as 'app' or 'application'
application = app
