"""Tuner params JSON load contract."""
import json
import tempfile
import unittest
from pathlib import Path

from trading_system.meta_learner.tuner import load_tuned_params


class TestTunerLoad(unittest.TestCase):
    def test_load_requires_policy_kwargs(self) -> None:
        payload = {
            "learning_rate": 1e-4,
            "n_steps": 512,
            "batch_size": 128,
            "policy_kwargs": {"pi": [64, 64], "vf": [64, 64]},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tuned.json"
            with open(p, "w") as f:
                json.dump(payload, f)
            loaded = load_tuned_params(path=p)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["learning_rate"], 1e-4)
            self.assertEqual(loaded["n_steps"], 512)
