"""Fictional company packs used for evaluation. They are not real certificates."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent / "synthetic"


def list_profiles() -> list[dict]:
    payload = json.loads((_ROOT / "profiles.json").read_text(encoding="utf-8"))
    return list(payload["profiles"])


def get_profile(profile_id: str) -> dict | None:
    for profile in list_profiles():
        if profile["id"] == profile_id:
            return profile
    return None


def profile_files(profile_id: str) -> list[tuple[str, bytes]]:
    profile = get_profile(profile_id)
    if profile is None:
        raise KeyError(profile_id)
    folder = _ROOT / profile["folder"]
    files: list[tuple[str, bytes]] = []
    for path in sorted(folder.glob("*.txt")):
        files.append((path.name, path.read_bytes()))
    if not files:
        raise FileNotFoundError(f"no synthetic documents for {profile_id}")
    return files
