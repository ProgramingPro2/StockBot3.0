"""Env registry sync, UI coverage, and enum validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from webapp.backend import env_manager


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text("# test env\n")
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(
        json.dumps(
            {
                "variables": [
                    {
                        "key": "LIVE_BUYING_POWER_MODE",
                        "default": "non_marginable",
                        "allowed_values": ["non_marginable", "buying_power"],
                        "group": "Trading",
                    },
                    {
                        "key": "TEST_WINDOW_PLACEMENT",
                        "default": "recent",
                        "allowed_values": ["recent", "oldest", "none"],
                        "group": "Walk-forward",
                    },
                    {
                        "key": "BOARDROOM_PORT",
                        "default": "28473",
                        "group": "Web",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(env_manager, "ENV_FILE", env_file)
    monkeypatch.setattr(env_manager, "ENV_REGISTRY_FILE", registry_file)
    env_manager.clear_registry_cache()
    return env_file


def test_ensure_defaults_writes_missing_keys(isolated_env: Path) -> None:
    assert env_manager.ensure_defaults_in_env() is True
    text = isolated_env.read_text()
    assert "LIVE_BUYING_POWER_MODE=non_marginable" in text
    assert "TEST_WINDOW_PLACEMENT=recent" in text
    assert "BOARDROOM_PORT=28473" in text
    assert env_manager.ensure_defaults_in_env() is False


def test_variables_for_ui_includes_registry_without_prior_env_keys(
    isolated_env: Path,
) -> None:
    env_manager.ensure_defaults_in_env()
    keys = {item["key"] for item in env_manager.variables_for_ui()}
    assert keys == {"LIVE_BUYING_POWER_MODE", "TEST_WINDOW_PLACEMENT", "BOARDROOM_PORT"}


def test_allowed_values_exposed_in_ui_payload(isolated_env: Path) -> None:
    env_manager.ensure_defaults_in_env()
    row = next(
        x
        for x in env_manager.variables_for_ui()
        if x["key"] == "LIVE_BUYING_POWER_MODE"
    )
    assert row["allowed_values"] == ["non_marginable", "buying_power"]
    assert row["value"] == "non_marginable"


def test_update_rejects_invalid_enum(isolated_env: Path) -> None:
    env_manager.ensure_defaults_in_env()
    with pytest.raises(env_manager.EnvValidationError, match="LIVE_BUYING_POWER_MODE"):
        env_manager.update({"LIVE_BUYING_POWER_MODE": "invalid"})


def test_update_accepts_valid_enum(isolated_env: Path) -> None:
    env_manager.ensure_defaults_in_env()
    env_manager.update({"LIVE_BUYING_POWER_MODE": "buying_power"})
    _, kv, _ = env_manager.load()
    assert kv["LIVE_BUYING_POWER_MODE"] == "buying_power"
