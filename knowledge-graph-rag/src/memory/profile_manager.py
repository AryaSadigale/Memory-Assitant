# FILE: src/memory/profile_manager.py
# CHANGES: Added @-prefix tolerant username login while preserving interactive profile selection.

import json
import os
import uuid
from dataclasses import dataclass
from typing import Optional

from loguru import logger

PROFILE_FILE = "/app/session_data/.profile"


@dataclass
class UserProfile:
    """Represents a registered user profile."""

    user_id: str
    username: str
    display_name: str
    created_at: str


class ProfileManager:
    """Handles profile creation, loading, and persistence across sessions."""

    def __init__(self, neo4j_client) -> None:
        """Initialize with Neo4j client for profile storage."""
        self.client = neo4j_client

    def _load_local_profile(self) -> Optional[UserProfile]:
        """
        Load profile from local file.
        Returns None if file does not exist or is invalid.
        """
        try:
            if not os.path.exists(PROFILE_FILE):
                return None
            with open(PROFILE_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return UserProfile(
                user_id=data["user_id"],
                username=data["username"],
                display_name=data["display_name"],
                created_at=data["created_at"],
            )
        except Exception as exc:
            logger.warning("Failed to load local profile: {}", exc)
            return None

    def _save_local_profile(self, profile: UserProfile) -> None:
        """Save profile to local file for persistence across restarts."""
        os.makedirs(os.path.dirname(PROFILE_FILE), exist_ok=True)
        with open(PROFILE_FILE, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "display_name": profile.display_name,
                    "created_at": profile.created_at,
                },
                handle,
                indent=2,
            )
        logger.debug("Profile saved to {}", PROFILE_FILE)

    async def _create_neo4j_profile(self, profile: UserProfile) -> None:
        """
        Create or update Profile node in Neo4j.
        Uses MERGE so reruns are safe.
        """
        try:
            await self.client.run_write(
                "MERGE (p:Profile {user_id: $user_id}) "
                "ON CREATE SET "
                "    p.username = $username, "
                "    p.display_name = $display_name, "
                "    p.created_at = $created_at, "
                "    p.chunk_count = 0, "
                "    p.document_count = 0, "
                "    p.memory_count = 0 "
                "ON MATCH SET "
                "    p.username = $username, "
                "    p.display_name = $display_name",
                {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "display_name": profile.display_name,
                    "created_at": profile.created_at,
                },
            )
            logger.debug("Profile node created/updated in Neo4j: {}", profile.username)
        except Exception as exc:
            logger.warning("Failed to create Neo4j profile: {}", exc)

    async def _load_neo4j_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from Neo4j by user_id."""
        try:
            result = await self.client.run_query(
                "MATCH (p:Profile {user_id: $user_id}) "
                "RETURN p.user_id AS user_id, "
                "       p.username AS username, "
                "       p.display_name AS display_name, "
                "       p.created_at AS created_at",
                {"user_id": user_id},
            )
            if result:
                row = result[0]
                return UserProfile(
                    user_id=row["user_id"],
                    username=row["username"],
                    display_name=row["display_name"],
                    created_at=row["created_at"],
                )
        except Exception as exc:
            logger.warning("Failed to load profile from Neo4j: {}", exc)
        return None

    async def _login_existing_user(self) -> "UserProfile":
        """
        Prompt for username and load that user's profile from Neo4j.
        Retries up to 3 times if username not found.
        """
        attempts = 0
        while attempts < 3:
            raw = input("\n  Enter your username: ").strip().lower()
            username = raw.lstrip("@")
            if not username:
                print("  Username cannot be empty.")
                attempts += 1
                continue
            try:
                result = await self.client.run_query(
                    "MATCH (p:Profile {username: $username}) "
                    "RETURN p.user_id AS user_id, "
                    "       p.username AS username, "
                    "       p.display_name AS display_name, "
                    "       p.created_at AS created_at",
                    {"username": username},
                )
                if result:
                    row = result[0]
                    profile = UserProfile(
                        user_id=row["user_id"],
                        username=row["username"],
                        display_name=row["display_name"],
                        created_at=row["created_at"],
                    )
                    self._save_local_profile(profile)
                    print(f"\n[Welcome back, {profile.display_name}!]")
                    return profile
                print(f"  Username '{username}' not found.")
                attempts += 1
            except Exception as exc:
                logger.error("Login lookup failed: {}", exc)
                attempts += 1

        print("  Too many failed attempts. Creating a new profile instead.")
        return await self._create_new_profile()

    async def _create_new_profile(self) -> "UserProfile":
        """
        Prompt for username and display name, create profile in Neo4j,
        save locally, and return the new UserProfile.
        """
        print()
        while True:
            username = input("  Choose a username: ").strip()
            if len(username) < 3:
                print("  Username must be at least 3 characters.")
                continue
            if not username.replace("_", "").replace("-", "").isalnum():
                print("  Only letters, numbers, - and _ are allowed.")
                continue
            try:
                existing = await self.client.run_query(
                    "MATCH (p:Profile {username: $username}) RETURN p.user_id AS uid",
                    {"username": username.lower()},
                )
                if existing:
                    print(f"  Username '{username}' is already taken. Choose another.")
                    continue
            except Exception:
                pass
            break

        display_name = input(
            "  Your display name (or press Enter to use username): "
        ).strip()
        if not display_name:
            display_name = username

        from datetime import datetime

        profile = UserProfile(
            user_id=str(uuid.uuid4()),
            username=username.lower(),
            display_name=display_name,
            created_at=datetime.utcnow().isoformat(),
        )
        await self._create_neo4j_profile(profile)
        self._save_local_profile(profile)

        print(f"\n[Profile created! Welcome, {display_name}.]")
        print(f"[Your user ID: {profile.user_id[:8]}...]")
        print()
        return profile

    async def get_or_create_profile(self) -> UserProfile:
        """
        Interactive profile selection shown on every startup.
        Always asks the user to choose: login, new, or continue.
        Only auto-continues if user presses Enter with no input
        and a previous profile exists.
        """
        local = self._load_local_profile()
        print("\n" + "=" * 50)
        print("  KNOWLEDGE GRAPH RAG - PROFILE")
        print("=" * 50)

        if local:
            print(f"\n  Last used: {local.display_name} (@{local.username})")
            print()
            print("  1. Continue as this user")
            print("  2. Login as a different user")
            print("  3. Create a new profile")
            print()
            choice = input("  Choose (1/2/3) or press Enter for 1: ").strip()
            if choice == "" or choice == "1":
                neo4j_profile = await self._load_neo4j_profile(local.user_id)
                if not neo4j_profile:
                    await self._create_neo4j_profile(local)
                    logger.info("Re-created Neo4j profile for {}", local.username)
                print(f"\n[Welcome back, {local.display_name}!]")
                return local
            if choice == "2":
                return await self._login_existing_user()
            if choice == "3":
                return await self._create_new_profile()
            print("  Invalid choice - continuing as last user.")
            print(f"\n[Welcome back, {local.display_name}!]")
            return local

        print()
        print("  1. Create a new profile")
        print("  2. Login as an existing user")
        print()
        choice = input("  Choose (1/2) or press Enter for 1: ").strip()
        if choice == "" or choice == "1":
            return await self._create_new_profile()
        if choice == "2":
            return await self._login_existing_user()
        return await self._create_new_profile()

    async def list_all_profiles(self) -> list:
        """List all profiles in the system. Admin use only."""
        try:
            return await self.client.run_query(
                "MATCH (p:Profile) "
                "RETURN p.username AS username, "
                "       p.display_name AS display_name, "
                "       p.chunk_count AS chunk_count, "
                "       p.created_at AS created_at "
                "ORDER BY p.created_at DESC"
            )
        except Exception as exc:
            logger.warning("Failed to list profiles: {}", exc)
            return []

    async def update_stats(
        self,
        user_id: str,
        chunk_delta: int = 0,
        doc_delta: int = 0,
        memory_delta: int = 0,
    ) -> None:
        """
        Increment usage stats on the Profile node.
        Called after ingestion and memory writes.
        Non-fatal if it fails.
        """
        try:
            await self.client.run_write(
                "MATCH (p:Profile {user_id: $user_id}) "
                "SET p.chunk_count = coalesce(p.chunk_count, 0) + $chunk_delta, "
                "    p.document_count = coalesce(p.document_count, 0) + $doc_delta, "
                "    p.memory_count = coalesce(p.memory_count, 0) + $memory_delta",
                {
                    "user_id": user_id,
                    "chunk_delta": chunk_delta,
                    "doc_delta": doc_delta,
                    "memory_delta": memory_delta,
                },
            )
        except Exception as exc:
            logger.debug("Stats update failed (non-fatal): {}", exc)
