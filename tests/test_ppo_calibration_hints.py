"""PPO calibration heuristics for web UI."""
from __future__ import annotations

from webapp.backend import system_info


def test_ppo_calibration_bounds() -> None:
    r = system_info.ppo_calibration_hints()
    assert 1 <= r.cpu_count <= 4096
    assert 1 <= r.recommended_n_envs <= 32
    assert 1 <= r.recommended_parallel_trials <= 8
    assert isinstance(r.suggest_torch_vec, bool)
    assert isinstance(r.cuda_available, bool)
    assert len(r.rationale) > 10
