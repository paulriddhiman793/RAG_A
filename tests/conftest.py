"""
Shared pytest configuration.
Adds the project root to sys.path so imports work without pip install -e.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
