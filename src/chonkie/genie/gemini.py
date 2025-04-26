"""Implementation of the GeminiGenie class."""
import importlib.util as importutil
import json
import os
from typing import Any, Optional

from .base import BaseGenie


class GeminiGenie(BaseGenie):
    """Gemini's Genie."""

    def __init__(self,
                model: str = "gemini-2.5-pro-preview-03-25",
                api_key: Optional[str] = None):
        """Initialize the GeminiGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use.

        """
        super().__init__()

        # Lazily import the dependencies
        self._import_dependencies()

        # Initialize the API key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GeminiGenie requires an API key. Either pass the `api_key` parameter or set the `GEMINI_API_KEY` in your environment.")

        # Initialize the client and model
        self.client = genai.Client(api_key=self.api_key) # type: ignore
        self.model = model

    #TODO: Fix the type hinting here later. 
    def generate(self, prompt: str, schema: Optional[Any] = None) -> Any:
        """Generate a response based on the given prompt."""
        if schema:
            response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': schema,
                }
            )
            return json.loads(response.text)
        else:
            response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
            )
            return response.text
    

    def _is_available(self) -> bool:
        """Check if all the dependencies are available in the environement."""
        if (importutil.find_spec("pydantic") is not None \
            and importutil.find_spec("google") is not None):
            return True
        return False

    def _import_dependencies(self) -> None:
        """Import all the required dependencies."""
        if self._is_available():
            global BaseModel, genai
            from google import genai
            from pydantic import BaseModel
        else:
            raise ImportError("One or more of the required modules are not available: [pydantic, google-genai]")