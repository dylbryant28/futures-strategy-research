"""Where market data is read from and results are written to.
Set FSR_DATA_DIR to your data folder; otherwise ./data next to src/ is used."""
import os
from pathlib import Path

DATA_DIR = os.environ.get("FSR_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")).rstrip("/\\") + os.sep
