import sys
from pathlib import Path

# Tests import `shared` and `agents` as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
