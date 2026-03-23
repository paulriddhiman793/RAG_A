"""
Run this script ONCE before main.py to diagnose import conflicts.
    python check_env.py
"""
import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
print(f"Project root : {PROJECT_ROOT}")
print(f"Python       : {sys.executable}")
print()

# Check sys.path order
print("=== sys.path (first 6 entries) ===")
for i, p in enumerate(sys.path[:6]):
    print(f"  [{i}] {p}")
print()

# Check where ingestion resolves to
try:
    import ingestion
    loc = getattr(ingestion, "__file__", None) or getattr(ingestion, "__path__", ["unknown"])
    print(f"ingestion    → {loc}")
except Exception as e:
    print(f"ingestion    → FAILED: {e}")

try:
    import ingestion.parsers as p
    loc = getattr(p, "__file__", None) or getattr(p, "__path__", ["unknown"])
    print(f"ingestion.parsers → {loc}")
    print(f"  exports: {dir(p)}")
except Exception as e:
    print(f"ingestion.parsers → FAILED: {e}")

print()

# Check for conflicting installed packages
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "pip", "show", "ingestion"],
    capture_output=True, text=True
)
if result.stdout:
    print("=== CONFLICT: 'ingestion' is an installed pip package! ===")
    print(result.stdout)
    print("Fix: pip uninstall ingestion")
else:
    print("No conflicting 'ingestion' pip package found (good).")