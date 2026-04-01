# Memory-Assistant: Knowledge Graph RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)

Memory-Assistant is a Neo4j-based Knowledge Graph RAG system that combines document ingestion, hybrid retrieval, personal memory, and multi-user profile isolation. The active application lives in `knowledge-graph-rag/`.

## What It Does

- Ingests PDF and text documents into a Neo4j-backed knowledge graph
- Stores personal memories separately from document knowledge
- Supports multiple user profiles with isolated private namespaces
- Uses hybrid retrieval with vector search plus BM25
- Supports document lookup, profile switching, and admin utilities

## Main Features

- Hybrid retrieval over vector search and BM25
- Document graph with `Document`, `Chunk`, `Entity`, and `Topic` nodes
- Personal memory storage and self-query support
- Multi-user profile isolation with per-user `user_id` stamping
- Terminal commands like `/ingest`, `/documents`, `/profile`, `/switch`, and `/admin users`
- Docker-based local setup with Neo4j

## Repository Layout

```text
Memory-Assistant/
|-- README.md
`-- knowledge-graph-rag/
    |-- docker-compose.yml
    |-- Dockerfile
    |-- ingest.py
    |-- main.py
    |-- requirements.txt
    `-- src/
        |-- chat/
        |-- graph/
        |-- ingestion/
        |-- llm/
        |-- memory/
        `-- retrieval/
```

## Architecture Diagram

```mermaid
flowchart TD
    U["User (Terminal)"] --> APP["main.py"]
    APP --> CHAT["ChatSession"]
    APP --> INGESTCLI["ingest.py"]

    CHAT --> PROFILE["ProfileManager"]
    CHAT --> INTENT["IntentClassifier"]
    CHAT --> RETR["RetrievalService"]
    CHAT --> MEM["MemoryService"]
    CHAT --> LLM["LLMClient (Groq)"]

    INGESTCLI --> PIPE["IngestionPipeline"]
    PIPE --> PARSER["DocumentParser"]
    PIPE --> SPLIT["ChunkSplitter"]
    PIPE --> EMB["Embedder"]
    PIPE --> NER["EntityExtractor"]

    RETR --> VEC["VectorSearch"]
    RETR --> BM25["BM25Search"]
    RETR --> GRAPH["GraphTraversal"]
    RETR --> RANK["Hybrid Ranking"]

    CHAT --> CREPO["ChunkRepository"]
    CHAT --> MREPO["MemoryRepository"]
    CHAT --> PREPO["ProfileRepository"]
    PIPE --> CREPO
    PIPE --> EREPO["EntityRepository"]

    CREPO --> NEO["Neo4j"]
    EREPO --> NEO
    MREPO --> NEO
    PREPO --> NEO
    VEC --> NEO
    BM25 --> NEO
    GRAPH --> NEO
```

## System Workflow Diagram

```mermaid
flowchart LR
    A["User starts app"] --> B["Profile selection"]
    B --> C["Existing user or new profile"]
    C --> D["Load session state"]
    D --> E["User enters query or command"]
    E --> F{"Slash command?"}

    F -->|Yes| G["Command handler"]
    G --> G1["/ingest -> pipeline"]
    G --> G2["/documents -> list docs"]
    G --> G3["/profile -> user stats"]
    G --> G4["/switch -> change profile"]
    G --> G5["/admin -> admin actions"]

    F -->|No| H["Intent classification"]
    H --> I{"Intent"}
    I -->|memory_share| J["Extract and store memory"]
    I -->|self_query| K["Retrieve memories"]
    I -->|knowledge_query| L["Hybrid retrieval"]
    I -->|document_lookup| M["Filename lookup, then topic fallback"]
    I -->|chitchat| N["Direct LLM reply"]

    K --> O["Context assembly"]
    L --> O
    M --> O
    O --> P["LLM answer"]
    P --> Q["Print response in terminal"]
```

## Knowledge Graph Working Diagram

```mermaid
flowchart TD
    USER["User"] -->|HAS_MEMORY| MEM["Memory"]

    DOC["Document"] -->|CONTAINS| CH1["Chunk"]
    DOC -->|CONTAINS| CH2["Chunk"]
    DOC -->|CONTAINS| CH3["Chunk"]

    CH1 -->|MENTIONS| ENT1["Entity"]
    CH2 -->|MENTIONS| ENT1
    CH2 -->|MENTIONS| ENT2["Entity"]
    CH3 -->|MENTIONS| ENT3["Entity"]

    CH1 -->|ABOUT_TOPIC| TOP1["Topic"]
    CH2 -->|ABOUT_TOPIC| TOP2["Topic"]

    ENT1 -->|CO_OCCURS_WITH| ENT2
    ENT2 -->|CO_OCCURS_WITH| ENT3
```

## How Knowledge Is Stored

- Personal memory:
  `(:User)-[:HAS_MEMORY]->(:Memory)`
- Ingested document knowledge:
  `(:Document)-[:CONTAINS]->(:Chunk)`
- Chunk enrichment:
  `(:Chunk)-[:MENTIONS]->(:Entity)` and `(:Chunk)-[:ABOUT_TOPIC]->(:Topic)`
- Cross-entity connections:
  `(:Entity)-[:CO_OCCURS_WITH]->(:Entity)`
- Multi-user isolation:
  `Chunk`, `Document`, and `Memory` nodes are stamped with `user_id`

## Retrieval Flow

1. User sends a query in the terminal.
2. The system classifies the intent.
3. For knowledge queries, it runs vector search plus BM25.
4. Results are merged, deduplicated, and ranked.
5. Context is assembled from the top chunks.
6. The LLM answers strictly from retrieved context.

## Setup

From the repository root:

```powershell
cd .\knowledge-graph-rag
docker compose build app
docker compose up -d neo4j
```

Set your Groq API key in PowerShell:

```powershell
$env:GROQ_API_KEY="your_actual_key"
```

Run the chat application:

```powershell
docker compose run --rm app python main.py
```

Run standalone ingestion:

```powershell
docker compose run --rm app python ingest.py /app/data
```

## Common Commands

```text
/ingest <path>
/documents
/profile
/switch
/memories
/forget <text>
/stats
/debug <query>
/admin help
/admin users
/admin ingest <username> <path>
/quit
```

## Example Questions

- `Tell me about knowledge graphs`
- `Tell me about nontowered airports`
- `Tell me about Chapter 10 from The Fintech Book PDF.pdf`
- `What do you know about me?`
- `What papers do you have?`

## Notes

- Local cache, session data, and dataset folders are excluded from Git.
- Docker build context is trimmed with `.dockerignore`.
- The Mermaid diagrams above are written in GitHub-compatible syntax.
