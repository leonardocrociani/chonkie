"""BaseGenie is the base class for all genies."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseGenie(ABC):
  """Base class for all Genies."""

  @abstractmethod
  def generate(self, prompt: str, schema: Optional[Any] = None) -> str:
    """Generate a response based on the given prompt."""
    raise NotImplementedError