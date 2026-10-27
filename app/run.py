"""Run from any working directory: python /path/to/app/run.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
