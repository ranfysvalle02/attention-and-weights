"""Pytest root config — makes app.py / demo.py importable from tests/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
