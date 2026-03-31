# FILE: src/llm/intent_classifier.py
# CHANGES: Added document_lookup intent and updated routing instructions for document-centric queries.

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

    SYSTEM_PROMPT = """Classify the user message into exactly one intent:
- knowledge_query: user wants factual information about a topic, concept, or field
- memory_share: user is sharing personal facts about themselves
- self_query: user is asking about themselves or what you know about them
- document_lookup: user is asking about a specific paper, file, or document by name or ID, or asking to list what documents are available
- chitchat: casual talk, short replies, corrections, one-word answers

Examples of document_lookup:
  "tell me about 2403.04782v1.pdf"
  "what is in the scalable_ai paper"
  "what papers do you have"
  "list your documents"
  "show me the osdi paper"

Reply with ONLY the intent label. No explanation. No punctuation."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the classifier with an LLM client."""
        self.llm_client = llm_client

    async def classify(self, message: str) -> str:
        """Classify a user message into one of the supported intents."""
        response = await self.llm_client.complete(
            self.SYSTEM_PROMPT,
            message,
            max_tokens=10,
            temperature=0.0,
        )
        normalized = response.strip().lower()
        if normalized not in VALID_INTENTS:
            return "knowledge_query"
        return normalized

    def should_extract_memory(self, message: str, intent: str) -> bool:
        """Return True when a message qualifies for personal fact extraction."""
        return intent == "memory_share" and len(message.split()) >= 5 and not message.startswith("/")
