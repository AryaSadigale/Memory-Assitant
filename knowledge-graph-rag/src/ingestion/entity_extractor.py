# FILE: src/ingestion/entity_extractor.py
# PURPOSE: Extract normalized named entities from text using spaCy NER.

from typing import Dict, List
from uuid import NAMESPACE_URL, uuid5

import spacy

from src.graph_models import EntityNode

ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "FAC", "WORK_OF_ART"}


class EntityExtractor:
    """Extract named entities from text using spaCy."""

    def __init__(self) -> None:
        """Load the spaCy English NER pipeline."""
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000

    def extract(self, text: str) -> List[EntityNode]:
        """Extract, normalize, and deduplicate allowed entities from a single text."""
        doc = self.nlp(text)
        seen: Dict[str, EntityNode] = {}
        for ent in doc.ents:
            if ent.label_ not in ALLOWED_LABELS:
                continue
            name = ent.text.strip().lower()
            if len(name) < 3:
                continue
            if name in seen:
                continue
            seen[name] = EntityNode(
                entity_id=str(uuid5(NAMESPACE_URL, f"{ent.label_}:{name}")),
                name=name,
                label=ent.label_,
                mention_count=1,
            )
        return list(seen.values())

    def extract_batch(self, texts: List[str]) -> List[List[EntityNode]]:
        """Extract entities from multiple texts efficiently with nlp.pipe()."""
        results: List[List[EntityNode]] = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            seen: Dict[str, EntityNode] = {}
            for ent in doc.ents:
                if ent.label_ not in ALLOWED_LABELS:
                    continue
                name = ent.text.strip().lower()
                if len(name) < 3:
                    continue
                if name in seen:
                    continue
                seen[name] = EntityNode(
                    entity_id=str(uuid5(NAMESPACE_URL, f"{ent.label_}:{name}")),
                    name=name,
                    label=ent.label_,
                    mention_count=1,
                )
            results.append(list(seen.values()))
        return results
