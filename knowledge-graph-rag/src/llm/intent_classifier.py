# FILE: src/llm/intent_classifier.py
# CHANGES: Tightened classifier guidance so topic queries stay in knowledge_query and only explicit file references use document_lookup.

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

    SYSTEM_PROMPT = """Classify the user message into exactly one intent.

- knowledge_query: user wants factual information about a topic, concept,
  technology, process, or field. This includes "tell me about X" where X
  is a topic, not a specific file.

- document_lookup: user is referencing a SPECIFIC document by filename,
  arxiv ID, or asking to list available documents. Only use this when
  the message contains a filename (ends in .pdf, .txt), an arxiv-style ID
  like 2403.04782, or explicit phrases like "what papers do you have",
  "list documents", "list files", "what files", "show documents".

- memory_share: user is sharing personal facts about themselves.

- self_query: user is asking what you know about them personally.

- chitchat: casual talk, greetings, short replies, corrections.

CRITICAL RULE: "Tell me about [topic]" is knowledge_query unless the
topic is a specific filename or document ID. Topics like "airports",
"machine learning", "chapter 10", "towered airport", "fintech" are
knowledge_query, NOT document_lookup.

Examples of knowledge_query:
  "Tell me about towered airports"
  "Explain aeronautical charts"
  "What is federated learning"
  "Tell me about chapter 10"
  "What does the book say about payments"

Examples of document_lookup:
  "Tell me about 2403.04782v1.pdf"
  "What is in 3445.pdf"
  "Tell me about the osdi paper"
  "What papers do you have"
  "List my documents"
  "Tell me about chapter 10 from 3445.pdf"

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
