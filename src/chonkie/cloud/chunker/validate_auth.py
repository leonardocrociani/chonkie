"""Cloud Token Chunking for Chonkie API."""

import os
from typing import Optional

import requests


class ValidateAuth:
    """Validate API key for Chonkie Cloud."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize ValidateAuth."""
        # If no API key is provided, use the environment variable
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the ValidateAuth constructor."
            )

        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200: 
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai or raise an issue on GitHub."
            )

        # Assign all the attributes to the instance
        self.api_key = api_key

        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai or raise an issue on GitHub."
            )

    def validate(self) -> bool:
        """Validate the API key."""
        # Define the payload for the request
        payload = {
            "api_key": self.api_key,
        }
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/validate/auth",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        # Check if the response is successful
        if response.status_code != 200:
            raise ValueError(
                f"Error from the Chonkie API: {response.status_code} {response.text}"
            )
        return True

    def __call__(self) -> bool:
        """Call the validate method."""
        return self.validate()
