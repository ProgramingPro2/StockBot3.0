"""Per-command Typer modules for the Boardroom CLI.

Each module defines a small typed function that builds an `argparse.Namespace`
and delegates to the implementation in :mod:`trading_system.cli.run`. This
keeps the existing `cmd_*` functions (and tests that import them) untouched
while giving us a modern Typer-based UX.
"""
