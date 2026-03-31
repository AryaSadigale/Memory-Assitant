# FILE: src/llm/llm_client.py
# PURPOSE: Wrap the Groq chat completion API behind a small async client.

from groq import AsyncGroq


class LLMClient:
    """Thin async wrapper around the Groq chat completion API."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the Groq async client."""
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Generate a chat completion from system and user prompts."""
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = response.choices[0].message.content or ""
        return message.strip()
