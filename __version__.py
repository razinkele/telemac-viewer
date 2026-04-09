"""Single-source version — reads from the VERSION file at package root."""
from pathlib import Path

VERSION_FILE = Path(__file__).parent / "VERSION"
__version__ = VERSION_FILE.read_text().strip()
