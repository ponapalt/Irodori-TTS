"""Search/filter helpers and favorite/recent persistence for tone presets.

Kept dependency-free (stdlib only) so it can be unit tested without importing
gradio or torch.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tone_presets import TonePreset, enabled_presets

DEFAULT_MAX_RECENT = 10


def search_presets(
    query: str = "",
    category: str | None = None,
    favorite_ids: set[str] | None = None,
    favorites_only: bool = False,
    presets: list[TonePreset] | None = None,
) -> list[TonePreset]:
    """Filter presets by free-text query, category id, and/or favorites.

    ``query`` matches (case-insensitively) against the preset name, id, and
    description. Adding a new preset to TONE_PRESETS makes it searchable here
    automatically -- there is no separate list to maintain.
    """
    items = enabled_presets(presets)
    if category:
        items = [p for p in items if p.category == category]
    if favorites_only:
        favs = favorite_ids or set()
        items = [p for p in items if p.id in favs]
    q = (query or "").strip().lower()
    if q:
        items = [
            p
            for p in items
            if q in p.name.lower() or q in p.description.lower() or q in p.id.lower()
        ]
    return items


class ToneLibraryStore:
    """Persists favorite / recently-used preset ids to a small JSON file.

    Mirrors the existing ref_audio_presets on-disk pattern used elsewhere in
    this app (plain files under the repo directory), just for preset ids
    instead of audio.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"favorites": [], "recent": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"favorites": [], "recent": []}
        if not isinstance(data, dict):
            return {"favorites": [], "recent": []}
        data.setdefault("favorites", [])
        data.setdefault("recent", [])
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def favorites(self) -> list[str]:
        return list(self._load()["favorites"])

    def recent(self) -> list[str]:
        return list(self._load()["recent"])

    def is_favorite(self, preset_id: str) -> bool:
        return preset_id in self._load()["favorites"]

    def toggle_favorite(self, preset_id: str) -> bool:
        """Toggle favorite state for preset_id. Returns the new state (True=favorited)."""
        data = self._load()
        favs = data["favorites"]
        if preset_id in favs:
            favs.remove(preset_id)
            is_fav = False
        else:
            favs.append(preset_id)
            is_fav = True
        self._save(data)
        return is_fav

    def push_recent(self, preset_id: str, max_len: int = DEFAULT_MAX_RECENT) -> list[str]:
        """Record preset_id as most-recently-used. Returns the updated recent list."""
        data = self._load()
        recent = [r for r in data["recent"] if r != preset_id]
        recent.insert(0, preset_id)
        data["recent"] = recent[:max_len]
        self._save(data)
        return list(data["recent"])
