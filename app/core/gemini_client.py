"""
Gemini Client

Wrapper for Vertex AI Gemini API calls.
Handles initialization, error handling, and fallback for missing credentials.
Provides a global singleton for use across the application.
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around Vertex AI Gemini generative model.

    Initializes Vertex AI on construction.  If credentials are missing or
    the project is not configured, the client gracefully degrades and
    ``is_available`` returns False.
    """

    def __init__(self) -> None:
        self.model = None
        self._initialized = False
        self._initialize()

    def _resolve_project(self) -> str | None:
        """Return the GCP project ID from settings or credentials file."""
        if settings.GOOGLE_CLOUD_PROJECT:
            return settings.GOOGLE_CLOUD_PROJECT
        # Fall back: read project_id from the service-account JSON
        import json, os
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if creds_path and os.path.isfile(creds_path):
            with open(creds_path) as f:
                return json.load(f).get("project_id")
        return None

    def _initialize(self) -> None:
        """Attempt to initialise the Vertex AI SDK and load the model."""
        project = self._resolve_project()
        if not project:
            logger.warning(
                "GCP project not found - Gemini running in demo mode"
            )
            return

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(
                project=project,
                location=settings.GOOGLE_CLOUD_LOCATION,
            )
            self.model = GenerativeModel(
                settings.GEMINI_MODEL,
            )
            self._initialized = True
            logger.info("Gemini client initialized successfully")
        except Exception as exc:
            logger.warning("Gemini initialization failed: %s", exc)
            logger.warning(
                "Running in demo mode - configure Google Cloud credentials "
                "for full functionality"
            )
            self._initialized = False

    @property
    def is_available(self) -> bool:
        """Return True if Gemini is ready to accept requests."""
        return self._initialized

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> str | None:
        """Generate a text response from the model.

        Args:
            system_prompt: The system-level instruction.
            user_prompt: The user-level input.
            temperature: Sampling temperature (default 0.1 for determinism).
            max_output_tokens: Maximum tokens in the response.

        Returns:
            The generated text, or None if the client is unavailable.
        """
        if not self._initialized:
            return None

        from vertexai.generative_models import Content, GenerativeModel, Part

        # Rebuild model with system instruction for this call
        model = GenerativeModel(
            settings.GEMINI_MODEL,
            system_instruction=system_prompt,
        )

        response = await model.generate_content_async(
            contents=[
                Content(role="user", parts=[Part.from_text(user_prompt)])
            ],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        return response.text

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list,
    ) -> object | None:
        """Generate a response that may include function calls.

        Args:
            prompt: The user prompt.
            tools: List of tool declarations (Vertex AI format).

        Returns:
            The model response object, or None if unavailable.
        """
        if not self._initialized:
            return None

        from vertexai.generative_models import Content, GenerativeModel, Part

        model = GenerativeModel(
            settings.GEMINI_MODEL,
            tools=tools,
        )
        response = await model.generate_content_async(
            contents=[
                Content(role="user", parts=[Part.from_text(prompt)])
            ],
            generation_config={"temperature": 0.0},
        )
        return response


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
gemini_client = GeminiClient()
