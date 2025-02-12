# orca_sindo/__init__.py
"""
ORCA to SINDO Interface
Provides functions to convert ORCA output files into SINDO input format.
"""

from .converter import convert
from .utils import read_orca_output, write_sindo_input

__all__ = ["xyz","engrad","hess","out"]
