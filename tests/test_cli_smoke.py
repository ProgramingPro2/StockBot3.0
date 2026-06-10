"""Smoke tests: CLI imports correctly and --help works for all major subcommands."""
from __future__ import annotations

import subprocess
import sys
import unittest


PYTHON = sys.executable
COMMANDS = [
    ["train", "--help"],
    ["tune", "--help"],
    ["backtest", "--help"],
    ["paper", "--help"],
    ["live", "--help"],
]


class TestCLIHelp(unittest.TestCase):
    """Each subcommand must respond to --help without error (no real execution)."""

    def _run_help(self, args: list[str]) -> tuple[int, str, str]:
        result = subprocess.run(
            [PYTHON, "-m", "trading_system"] + args,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr

    def test_module_help(self) -> None:
        rc, out, err = self._run_help(["--help"])
        self.assertEqual(rc, 0, f"trading_system --help failed:\n{err}")
        self.assertIn("usage", (out + err).lower())

    def test_train_help(self) -> None:
        rc, out, err = self._run_help(["train", "--help"])
        self.assertEqual(rc, 0, f"train --help failed:\n{err}")

    def test_tune_help(self) -> None:
        rc, out, err = self._run_help(["tune", "--help"])
        self.assertEqual(rc, 0, f"tune --help failed:\n{err}")

    def test_backtest_help(self) -> None:
        rc, out, err = self._run_help(["backtest", "--help"])
        self.assertEqual(rc, 0, f"backtest --help failed:\n{err}")

    def test_paper_help(self) -> None:
        rc, out, err = self._run_help(["paper", "--help"])
        self.assertEqual(rc, 0, f"paper --help failed:\n{err}")

    def test_live_help(self) -> None:
        rc, out, err = self._run_help(["live", "--help"])
        self.assertEqual(rc, 0, f"live --help failed:\n{err}")


class TestCLIImport(unittest.TestCase):
    """trading_system package must import cleanly without side effects."""

    def test_package_imports(self) -> None:
        result = subprocess.run(
            [PYTHON, "-c", "import trading_system; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"Import failed:\n{result.stderr}")
        self.assertIn("OK", result.stdout)

    def test_cli_run_module_imports(self) -> None:
        result = subprocess.run(
            [PYTHON, "-c", "from trading_system.cli.run import _execute_rebalance_passes; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"Import failed:\n{result.stderr}")
        self.assertIn("OK", result.stdout)
