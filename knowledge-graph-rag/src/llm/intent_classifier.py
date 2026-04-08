# FILE: src/llm/intent_classifier.py
# CHANGES: Replaced the classifier prompt so introductions and short replies route to the correct intents.

from src.llm.llm_client import LLMClient

VALID_INTENTS = {
    "knowledge_query",
    "memory_share",
    "self_query",
    "chitchat",
    "document_lookup",
}


class IntentClassifier:
    """LLM-backed intent classifier for routing chat requests."""

    SYSTEM_PROMPT = """You are an intent classifier for a personal
memory assistant. Classify the message into exactly one intent.

memory_share - user is telling you personal facts about themselves:
  name, job, location, preferences, relationships, experiences.
  Use this even if the message is casual or mixed with other content.
  Examples: "Hello my name is Arya I work in aviation"
            "I am the CEO of a fintech company"
            "I live in Boston and work at HEBBRIX"
            "I love flying A380s since 2015"

self_query - user is asking what YOU know about THEM personally:
  Examples: "Tell me about myself"
            "What is my name?"
            "Where do I work?"
            "Which airlines did I fly?"
            "What are my preferences?"

knowledge_query - user wants facts about the world, a topic, or field:
  Examples: "What is a taxiway?"
            "Tell me about fintech"
            "Explain neural networks"
            "Why is federated learning useful?"

document_lookup - user asks about a specific file or paper by name:
  Examples: "Tell me about 3445.pdf"
            "What papers do you have?"
            "List documents"

chitchat - casual talk, short replies, greetings, acknowledgements:
  Examples: "Cool" "okay" "Thanks" "Got it" "Hi" "Yes" "Nope"

CRITICAL RULES:
1. If message contains "my name is", "I work at", "I live in",
   "I am a", "I love", "I prefer", it is ALWAYS memory_share.
2. Words like "Cool", "okay", "sure", "thanks", "hi", "hello"
   alone or with only 1-2 other words = ALWAYS chitchat.
3. Never return knowledge_query for personal introductions.
4. Reply with ONLY the intent label. Nothing else."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the classifier with an LLM client."""
        self.llm_client = llm_client

    async def classify(self, message: str) -> str:
        """Classify a user message into one of the supported intents."""
        response = await self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            temperature=0.0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )
        content = (response.choices[0].message.content or "").strip().lower()
        first_line = content.splitlines()[0] if content else ""
        normalized = first_line.strip(" .")
        if normalized not in VALID_INTENTS:
            return "knowledge_query"
        return normalized

    def should_extract_memory(self, message: str, intent: str) -> bool:
        """Return True when a message qualifies for personal fact extraction."""
        return intent == "memory_share" and len(message.split()) >= 5 and not message.startswith("/")
