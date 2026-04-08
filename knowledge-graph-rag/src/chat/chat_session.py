# FILE: src/chat/chat_session.py
# CHANGES: Updated /debug to use retrieval settings and show final hybrid-ranked chunks.

import asyncio
import os
import re
from typing import Optional

from loguru import logger

from src.graph.chunk_repository import ChunkRepository
from src.graph.memory_repository import MemoryRepository
from src.graph.neo4j_client import Neo4jClient
from src.graph.profile_repository import ProfileRepository
from src.ingestion.pipeline import IngestionPipeline
from src.llm.context_assembler import ContextAssembler
from src.llm.intent_classifier import IntentClassifier
from src.llm.llm_client import LLMClient
from src.memory.memory_service import MemoryService
from src.memory.profile_manager import ProfileManager
from src.retrieval.retrieval_service import RetrievalService


class ChatSession:
    """Terminal session logic for chat, memory, and ingestion commands."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        pipeline: IngestionPipeline,
        retrieval: RetrievalService,
        memory_service: MemoryService,
        memory_repo: MemoryRepository,
        classifier: IntentClassifier,
        assembler: ContextAssembler,
        llm: LLMClient,
        user_id: str,
        chunk_repo: Optional[ChunkRepository] = None,
        username: str = "default",
        profile_repo: Optional[ProfileRepository] = None,
        profile_manager: Optional[ProfileManager] = None,
    ) -> None:
        """Store chat dependencies for a single interactive session."""
        self.neo4j_client = neo4j_client
        self.pipeline = pipeline
        self.retrieval = retrieval
        self.memory_service = memory_service
        self.memory_repo = memory_repo
        self.classifier = classifier
        self.assembler = assembler
        self.llm = llm
        self.user_id = user_id
        self.chunk_repo = chunk_repo
        self.username = username
        self.profile_repo = profile_repo
        self.profile_manager = profile_manager
        self.simple_system = (
            "You are a friendly, concise chat assistant. "
            "Reply in 1-2 natural sentences. "
            "No markdown, no headers, no bullet points."
        )

    async def _complete_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        """Call the model directly so response formatting follows the supplied prompt."""
        response = await self.llm.client.chat.completions.create(
            model=self.llm.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def _read_input(self) -> str:
        """Read user input without blocking the event loop."""
        return await asyncio.to_thread(input, "You: ")

    async def _handle_ingest(self, path: str) -> None:
        """Handle the /ingest command for files or directories."""
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            print(f"[Path not found: {resolved}]")
            return
        if os.path.isdir(resolved):
            await self.pipeline.ingest_directory(resolved, user_id=self.user_id)
            if self.profile_manager is not None:
                await self.profile_manager.update_stats(self.user_id, doc_delta=0)
            return
        result = await self.pipeline.ingest_file(resolved, user_id=self.user_id)
        if self.profile_manager is not None:
            await self.profile_manager.update_stats(
                self.user_id,
                chunk_delta=result.chunk_count,
                doc_delta=0 if result.skipped else 1,
            )

    async def _handle_memories(self) -> None:
        """Display stored memories for the active user."""
        memories = await self.memory_repo.list_recent(self.user_id, limit=50)
        if not memories:
            print("[No stored memories]\n")
            return
        print("Stored memories:")
        for index, memory in enumerate(memories, start=1):
            print(f"  {index}. {memory.content}")
        print()

    async def _handle_forget(self, text: str) -> None:
        """Delete the first stored memory matching the provided text."""
        if not text:
            print("[Usage: /forget <text>]\n")
            return
        matches = await self.memory_repo.find_by_text(self.user_id, text, limit=10)
        if not matches:
            print("[No matching memory found]\n")
            return
        await self.memory_repo.delete(matches[0].memory_id)
        print(f"[Forgot memory: {matches[0].content}]\n")

    async def _handle_stats(self) -> None:
        """Display current graph statistics for the active user."""
        query = """
        CALL {
            MATCH (c:Chunk {user_id: $user_id}) RETURN count(c) AS chunk_count
        }
        CALL {
            MATCH (e:Entity) RETURN count(e) AS entity_count
        }
        CALL {
            MATCH (t:Topic) RETURN count(t) AS topic_count
        }
        CALL {
            MATCH (m:Memory {user_id: $user_id}) RETURN count(m) AS memory_count
        }
        RETURN chunk_count, entity_count, topic_count, memory_count
        """
        rows = await self.neo4j_client.run_query(query, {"user_id": self.user_id})
        stats = rows[0] if rows else {}
        print(
            f"[Chunks: {stats.get('chunk_count', 0)} | "
            f"Entities: {stats.get('entity_count', 0)} | "
            f"Topics: {stats.get('topic_count', 0)} | "
            f"Memories: {stats.get('memory_count', 0)}]\n"
        )

    async def _handle_documents(self) -> None:
        """Display the ingested document catalog."""
        if self.chunk_repo is None:
            print("Document listing is unavailable.\n")
            return
        docs = await self.chunk_repo.list_all_documents(user_id=self.user_id)
        if docs:
            print(f"\n{'=' * 50}")
            print(f"INGESTED DOCUMENTS ({len(docs)} total)")
            print(f"{'=' * 50}")
            for document in docs:
                title = document.get("title") or document["filename"]
                print(f"  {title}")
                print(f"    File: {document['filename']}")
                print(f"    Chunks: {document['chunk_count']} | Pages: {document['page_count']}")
            print(f"{'=' * 50}\n")
        else:
            print("No documents ingested yet. Use /ingest <path>.\n")

    async def _handle_profile(self) -> None:
        """Display the current profile and storage stats."""
        if self.profile_repo is None:
            print("Profile repository is unavailable.\n")
            return
        stats = await self.profile_repo.get_user_stats(self.user_id)
        print(f"\n{'=' * 50}")
        print("YOUR PROFILE")
        print(f"{'=' * 50}")
        print(f"  Username:    {self.username}")
        print(f"  User ID:     {self.user_id[:8]}...")
        print(f"  Documents:   {stats['documents']}")
        print(f"  Chunks:      {stats['chunks']}")
        print(f"  Memories:    {stats['memories']}")
        print(f"{'=' * 50}\n")

    async def _handle_debug(self, debug_query: str) -> None:
        """Run retrieval diagnostics for a query."""
        if not debug_query:
            print("[Usage: /debug <query>]\n")
            return
        print(f"\n[DEBUG] Running retrieval for: '{debug_query}'")
        embedding = self.retrieval.embedder.embed_text(debug_query)
        print(f"[DEBUG] Embedding dim: {len(embedding)}")

        v_hits = await self.retrieval.vector_search.search_chunks(
            embedding,
            top_k=self.retrieval.settings.vector_top_k,
            user_id=self.user_id,
        )
        print(f"[DEBUG] Vector hits: {len(v_hits)}")
        for hit in v_hits[:3]:
            print(f"  score={hit.score:.3f} | {hit.source_file} | {hit.content[:60]}...")

        b_hits = await self.retrieval.bm25_search.search_chunks(
            debug_query,
            top_k=self.retrieval.settings.bm25_top_k,
            user_id=self.user_id,
        )
        print(f"[DEBUG] BM25 hits: {len(b_hits)}")
        for hit in b_hits[:3]:
            print(f"  score={hit.score:.3f} | {hit.source_file} | {hit.content[:60]}...")
        graph_hits = await self.retrieval.graph_traversal.expand_from_chunks(
            [hit.id for hit in v_hits[:3]],
            user_id=self.user_id,
        )
        final_hits = self.retrieval.hybrid_ranker.rank(
            v_hits,
            b_hits,
            graph_hits,
            top_k=self.retrieval.settings.final_top_k,
        )
        print(f"[DEBUG] After ranking: {len(final_hits)} final chunks")
        for hit in final_hits:
            print(
                f"  final_score={hit.score:.3f} | "
                f"{hit.source_file} p{hit.page_number} | "
                f"{hit.content[:60]}..."
            )
        print()

    def _extract_filename_from_query(self, text: str) -> str:
        """
        Extract a filename from user input.
        Handles filenames with spaces, dots, and mixed case.
        Returns empty string if no filename found.
        """
        from_pattern = re.compile(
            r"(?:from|in|of|about)\s+([\w][\w\s\-\.]*?\.(pdf|txt|md))",
            re.IGNORECASE
        )
        from_match = from_pattern.search(text)
        if from_match:
            return from_match.group(1).strip()

        arxiv_pattern = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
        arxiv_match = arxiv_pattern.search(text)
        if arxiv_match:
            return arxiv_match.group(0)

        simple_pattern = re.compile(
            r"\b([\w\-\.]+\.(pdf|txt|md))\b", re.IGNORECASE
        )
        simple_match = simple_pattern.search(text)
        if simple_match:
            return simple_match.group(1)

        return ""

    def _print_admin_help(self) -> None:
        """Print admin command reference."""
        print()
        print("  Admin commands:")
        print("  /admin users              list all registered profiles")
        print("  /admin ingest <u> <path>  ingest files as another user")
        print("  /admin help               show this help")
        print()

    async def _handle_admin_users(self) -> None:
        """
        List all profiles registered in the system.
        Queries Neo4j directly for live data.
        """
        try:
            rows = await self.neo4j_client.run_query(
                "MATCH (p:Profile) "
                "OPTIONAL MATCH (c:Chunk {user_id: p.user_id}) "
                "OPTIONAL MATCH (m:Memory {user_id: p.user_id}) "
                "OPTIONAL MATCH (d:Document {user_id: p.user_id}) "
                "RETURN p.username AS username, "
                "       p.display_name AS display_name, "
                "       p.user_id AS user_id, "
                "       p.created_at AS created_at, "
                "       count(DISTINCT c) AS chunks, "
                "       count(DISTINCT m) AS memories, "
                "       count(DISTINCT d) AS documents "
                "ORDER BY p.created_at ASC"
            )
            if not rows:
                print("\n[No profiles found]\n")
                return
            print(f"\n{'=' * 60}")
            print(f"  ALL PROFILES ({len(rows)} registered users)")
            print(f"{'=' * 60}")
            for row in rows:
                marker = " <- you" if row["user_id"] == self.user_id else ""
                print(f"  @{row['username']} - {row['display_name']}{marker}")
                print(
                    f"    ID: {row['user_id'][:8]}...  "
                    f"Docs: {row['documents']}  "
                    f"Chunks: {row['chunks']}  "
                    f"Memories: {row['memories']}"
                )
            print(f"{'=' * 60}\n")
        except Exception as exc:
            logger.error("Admin users listing failed: {}", exc)
            print("[Failed to list users]\n")

    async def _handle_admin_ingest(self, target_username: str, path: str) -> None:
        """
        Ingest files into another user's namespace.
        Resolves username to user_id then calls the pipeline.
        """
        try:
            result = await self.neo4j_client.run_query(
                "MATCH (p:Profile {username: $username}) "
                "RETURN p.user_id AS user_id, p.display_name AS display_name",
                {"username": target_username.lower()},
            )
            if not result:
                print(f"[User '{target_username}' not found]\n")
                return
            target_user_id = result[0]["user_id"]
            display = result[0]["display_name"]
            print(f"[Ingesting into {display}'s namespace (@{target_username})]")
            resolved = os.path.abspath(path)
            if not os.path.exists(resolved):
                print(f"[Path not found: {resolved}]\n")
                return
            if os.path.isdir(resolved):
                await self.pipeline.ingest_directory(resolved, user_id=target_user_id)
            else:
                await self.pipeline.ingest_file(resolved, user_id=target_user_id)
            print(f"[Ingestion complete for @{target_username}]\n")
        except Exception as exc:
            logger.error("Admin ingest failed: {}", exc)
            print(f"[Admin ingest failed: {exc}]\n")

    async def _handle_switch(self) -> None:
        """
        Switch the active user mid-session without restarting.
        Prompts for login or new profile, updates session state.
        """
        if self.profile_manager is None:
            print("[Profile manager not available]\n")
            return
        print("\n[Switching user - current session will end for this user]")
        new_profile = await self.profile_manager.get_or_create_profile()
        self.user_id = new_profile.user_id
        self.username = new_profile.username
        memories = await self.memory_service.load_for_session(self.user_id, limit=30)
        if memories:
            print(f"[Loaded {len(memories)} memories for {new_profile.display_name}]")
        print(f"[Now chatting as: {new_profile.display_name}]\n")

    async def _handle_command(self, user_input: str) -> Optional[bool]:
        """Handle slash commands and indicate whether the loop should continue."""
        if user_input.startswith("/ingest "):
            await self._handle_ingest(user_input[len("/ingest ") :].strip())
            return True
        if user_input == "/memories":
            await self._handle_memories()
            return True
        if user_input.startswith("/forget "):
            await self._handle_forget(user_input[len("/forget ") :].strip())
            return True
        if user_input == "/stats":
            await self._handle_stats()
            return True
        if user_input.lower() == "/documents":
            await self._handle_documents()
            return True
        if user_input.lower() == "/profile":
            await self._handle_profile()
            return True
        if user_input.lower().startswith("/debug "):
            await self._handle_debug(user_input[7:].strip())
            return True
        if user_input.lower() == "/admin users":
            await self._handle_admin_users()
            return True
        if user_input.lower().startswith("/admin ingest "):
            parts = user_input.split(" ", 3)
            if len(parts) == 4:
                target_user = parts[2]
                path = parts[3]
                await self._handle_admin_ingest(target_user, path)
            else:
                print("[Usage: /admin ingest <username> <path>]\n")
            return True
        if user_input.lower() == "/admin help":
            self._print_admin_help()
            return True
        if user_input.lower() == "/switch":
            await self._handle_switch()
            return True
        if user_input == "/quit":
            return False
        if user_input.startswith("/"):
            print("[Unknown command]\n")
            return True
        return None

    async def run(self) -> None:
        """Run the interactive chat loop until the user exits."""
        print("Commands:")
        print("  /ingest <path>    ingest a file or directory")
        print("  /memories         show stored memories")
        print("  /forget <text>    delete a memory containing this text")
        print("  /stats            show graph statistics")
        print("  /documents        list ingested documents")
        print("  /profile          show your profile and storage stats")
        print("  /debug <query>    inspect vector and BM25 retrieval")
        print("  /switch           switch to a different user")
        print("  /admin help       admin commands (manage users)")
        print("  /quit             exit\n")
        while True:
            user_input = (await self._read_input()).strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                should_continue = await self._handle_command(user_input)
                if should_continue is False:
                    break
                if should_continue:
                    continue

            stripped_input = user_input.strip()
            if len(stripped_input.split()) <= 2 and "?" not in stripped_input:
                response = await self._complete_raw(
                    self.simple_system,
                    stripped_input,
                    max_tokens=60,
                    temperature=0.7,
                )
                print(f"AI: {response}\n")
                continue

            intent = await self.classifier.classify(user_input)
            print(f"  [{intent}]")

            if intent == "memory_share":
                created = await self.memory_service.process_and_store(user_input, self.user_id)
                if created and self.profile_manager is not None:
                    await self.profile_manager.update_stats(self.user_id, memory_delta=len(created))
                response = await self._complete_raw(
                    "You are a concise assistant acknowledging that a memory was stored.",
                    "Acknowledge the memory briefly in one sentence.",
                    max_tokens=40,
                    temperature=0.2,
                )
            elif intent == "self_query":
                hits = await self.retrieval.retrieve_memory(user_input, self.user_id)
                packet = self.assembler.build_memory_context(user_input, hits)
                if hits:
                    await self.memory_repo.touch_many([hit.id for hit in hits])
                response = await self._complete_raw(
                    packet.system_prompt,
                    packet.user_prompt,
                    max_tokens=600,
                    temperature=0.0,
                )
            elif intent == "document_lookup":
                extracted_filename = self._extract_filename_from_query(user_input)
                hits = []
                if extracted_filename:
                    hits = await self.retrieval.retrieve_by_filename(
                        extracted_filename, user_id=self.user_id
                    )

                if not hits:
                    topic_query = re.sub(
                        r"(?:from|in|of|about)\s+[\w\s\-\.]*?\.(pdf|txt|md)",
                        "",
                        user_input,
                        flags=re.IGNORECASE
                    ).strip()
                    if len(topic_query.split()) >= 3:
                        logger.debug(
                            "document_lookup fallback to knowledge_query: '{}'",
                            topic_query
                        )
                        hits = await self.retrieval.retrieve_with_query_expansion(
                            topic_query, user_id=self.user_id
                        )

                if not hits:
                    if self.chunk_repo is not None:
                        docs = await self.chunk_repo.list_all_documents(
                            user_id=self.user_id
                        )
                        if docs:
                            doc_list = "\n".join([
                                f"  - {d['filename']} "
                                f"({d['chunk_count']} chunks, {d['page_count']} pages)"
                                for d in docs[:20]
                            ])
                            response = (
                                f"I have {len(docs)} document(s) ingested:\n{doc_list}"
                            )
                        else:
                            response = (
                                "No documents found. Use /ingest <path> to add documents."
                            )
                    else:
                        response = "No documents available."
                else:
                    packet = self.assembler.build_knowledge_context(user_input, hits)
                    response = await self._complete_raw(
                        packet.system_prompt,
                        packet.user_prompt,
                        max_tokens=1500,
                    )
            elif intent == "knowledge_query":
                hits = await self.retrieval.retrieve_with_query_expansion(user_input, user_id=self.user_id)
                packet = self.assembler.build_knowledge_context(user_input, hits)
                response = await self._complete_raw(
                    packet.system_prompt,
                    packet.user_prompt,
                    max_tokens=1500,
                )
            elif intent == "chitchat" or intent not in {
                "memory_share", "self_query",
                "knowledge_query", "document_lookup"
            }:
                response = await self._complete_raw(
                    self.simple_system,
                    user_input,
                    max_tokens=80,
                    temperature=0.7,
                )

            logger.debug("LLM response length: {}", len(response))
            print(f"AI: {response}\n")
