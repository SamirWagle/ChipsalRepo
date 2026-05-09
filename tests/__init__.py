"""
Test initialization file.

Makes tests/ a package and provides common test utilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


__version__ = "1.0.0"
__author__ = "Samir Thapa"
