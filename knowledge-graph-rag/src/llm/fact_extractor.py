# FILE: src/llm/fact_extractor.py
# PURPOSE: Extract user-stated personal facts from conversation turns.

from typing import List

from src.llm.llm_client import LLMClient


class FactExtractor:
    """Extract personal facts from user messages with the LLM."""

    SYSTEM_PROMPT = """Extract factual statements about the user from their message.
Return only statements the user made about themselves.
One fact per line. No bullet points. No numbering.
If no personal facts exist, return exactly: NONE"""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the fact extractor with an LLM client."""
        self.llm_client = llm_client

    async def extract(self, message: str) -> List[str]:
        """Extract factual self-statements from a message."""
        response = await self.llm_client.complete(
            self.SYSTEM_PROMPT,
            message,
            max_tokens=200,
            temperature=0.0,
        )
        if response.strip().upper() == "NONE":
            return []
        facts = []
        for line in response.splitlines():
            fact = line.strip().lstrip("-").strip()
            if len(fact) >= 10:
                facts.append(fact)
        return facts
