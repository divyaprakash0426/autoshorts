#!/usr/bin/env python3
"""
AutoShorts - Entry point script.
Run this from the project root to generate shorts.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shorts import main

if __name__ == "__main__":
    main()
