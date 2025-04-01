"""Custom types for recursive chunking."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union
from chonkie.types.base import Chunk

@dataclass
class RecursiveLevel:
    """Class to express chunking delimiters at different levels.
    
    Attributes:
        delimiters (List[str]): List of delimiters for the chunking level.
        level (int): The level of chunking."""