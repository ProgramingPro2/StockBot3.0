"""Env UI hint merge and coverage."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from webapp.backend import env_manager


def test_variables_for_ui_non_empty_descriptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row returned for the repo template has a non-empty description."""
    env_file = tmp_path / ".env"
    env_file.write_text("# empty for test\n")
    monkeypatch.setattr(env_manager, "ENV_FILE", env_file)
    env_manager.clear_env_hints_cache()
    env_manager._load_env_hints.cache_clear()

    for item in env_manager.variables_for_ui():
        desc = item.get("description")
        assert isinstance(desc, str) and desc.strip(), item["key"]


def test_user_comment_overrides_hints_and_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# my custom note for port\nBOARDROOM_PORT=9999\n",
    )
    monkeypatch.setattr(env_manager, "ENV_FILE", env_file)
    env_manager.clear_env_hints_cache()
    env_manager._load_env_hints.cache_clear()

    hints_path = tmp_path / "hints.json"
    hints_path.write_text(json.dumps({"BOARDROOM_PORT": "from json registry"}))
    monkeypatch.setattr(env_manager, "ENV_HINTS_FILE", hints_path)
    env_manager._load_env_hints.cache_clear()

    row = next(
        x for x in env_manager.variables_for_ui() if x["key"] == "BOARDROOM_PORT"
    )
    assert row["description"] == "my custom note for port"


def test_registry_overrides_example_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BOARDROOM_PORT=28473\n")
    monkeypatch.setattr(env_manager, "ENV_FILE", env_file)
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(json.dumps({"BOARDROOM_PORT": "registry wins"}))
    monkeypatch.setattr(env_manager, "ENV_HINTS_FILE", hints_path)
    env_manager.clear_env_hints_cache()
    env_manager._load_env_hints.cache_clear()

    row = next(
        x for x in env_manager.variables_for_ui() if x["key"] == "BOARDROOM_PORT"
    )
    assert row["description"] == "registry wins"


def test_underscore_json_keys_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BOARDROOM_PORT=1\n")
    monkeypatch.setattr(env_manager, "ENV_FILE", env_file)
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps({"_meta": "skip", "BOARDROOM_PORT": "from registry"}),
    )
    monkeypatch.setattr(env_manager, "ENV_HINTS_FILE", hints_path)
    env_manager.clear_env_hints_cache()
    env_manager._load_env_hints.cache_clear()

    row = next(
        x for x in env_manager.variables_for_ui() if x["key"] == "BOARDROOM_PORT"
    )
    assert row["description"] == "from registry"
