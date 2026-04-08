# FILE: src/llm/fact_extractor.py
# CHANGES: Expanded the extraction prompt so one message yields every personal fact stated.

from typing import List

from src.llm.llm_client import LLMClient


class FactExtractor:
    """Extract personal facts from user messages with the LLM."""

    SYSTEM_PROMPT = """Extract ALL factual statements about the user
from their message. Return every personal fact stated, one per line.

Include facts about:
- Name, age, gender
- Job, company, role, workplace
- Location, city, country, where they live or stay
- Relationships, family
- Preferences, hobbies, interests
- Experiences, achievements
- Any other personal information

Rules:
- Write each fact as a complete standalone sentence
- One fact per line, no bullets, no numbers
- If no personal facts exist, return exactly: NONE
- Do NOT miss any fact - extract everything stated
- Do NOT combine multiple facts into one line"""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the fact extractor with an LLM client."""
        self.llm_client = llm_client

    async def extract(self, message: str) -> List[str]:
        """Extract factual self-statements from a message."""
        response = await self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )
        content = response.choices[0].message.content or ""
        if content.strip().upper() == "NONE":
            return []
        facts = []
        for line in content.splitlines():
            fact = line.strip().lstrip("-").strip()
            if len(fact) >= 10:
                facts.append(fact)
        return facts
