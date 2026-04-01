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

        # 🔥 Adaptive + strong system prompt (universal for any PDF)
        strong_system_prompt = f"""
You are an expert assistant answering STRICTLY from provided context.

CRITICAL RULES:
- DO NOT summarize or compress information
- EXTRACT and EXPAND all important details from the context
- Cover ALL key points present in the context — do not skip anything
- If a list or multiple items are present → explain EACH item clearly
- Provide complete explanations, not short answers

STRUCTURE RULE:
- Adapt the answer structure based on the content
- If the topic is conceptual → use:
    Definition, Key Concepts, Explanation, Applications, Trends
- If the content is procedural → use:
    Steps / Process explanation
- If the content is descriptive → use:
    Detailed explanation with sections
- If the content contains lists/tables → explain each item clearly

GENERAL:
- Always use clear headings
- Maintain logical flow
- Combine information from all sources
- NEVER say "not enough information" if context exists

---

CONTEXT:
{system_prompt}
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": strong_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        message = response.choices[0].message.content or ""
        return message.strip()