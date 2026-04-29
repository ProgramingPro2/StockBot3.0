"""Tuner vec-env teardown after interrupt-style IPC failures."""
import unittest
from unittest.mock import MagicMock

from trading_system.meta_learner import tuner as tuner_mod
from trading_system.meta_learner.tuner import _safe_vec_env_close
from trading_system.utils import graceful_shutdown


class TestSafeVecEnvClose(unittest.TestCase):
    def tearDown(self) -> None:
        graceful_shutdown.reset_for_tests()

    def test_broken_pipe_does_not_propagate(self) -> None:
        env = MagicMock()
        env.close.side_effect = BrokenPipeError(32, "Broken pipe")
        _safe_vec_env_close(env)
        env.close.assert_called_once()

    def test_close_success(self) -> None:
        env = MagicMock()
        _safe_vec_env_close(env)
        env.close.assert_called_once()

    def test_oserror_errno_pipe_ignored(self) -> None:
        env = MagicMock()
        env.close.side_effect = OSError(32, "Broken pipe")
        _safe_vec_env_close(env)
        env.close.assert_called_once()

    def test_generic_exception_does_not_propagate(self) -> None:
        env = MagicMock()
        env.close.side_effect = RuntimeError("unexpected")
        _safe_vec_env_close(env)
        env.close.assert_called_once()

    def test_ipc_tuple_is_defined(self) -> None:
        self.assertIn(BrokenPipeError, tuner_mod._VEC_ENV_CLOSE_IPC_ERRORS)
        self.assertIn(EOFError, tuner_mod._VEC_ENV_CLOSE_IPC_ERRORS)
