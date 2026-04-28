"""`boardroom schedule` - scheduler daemon and systemd integration."""
from __future__ import annotations

from typing import Optional

import typer

from ...config import BASE_DIR
from ._common import make_args


def register(app: typer.Typer) -> None:
    schedule_app = typer.Typer(help="Run the scheduler daemon or manage systemd integration.", no_args_is_help=False)
    app.add_typer(schedule_app, name="schedule")

    default_schedule = str(BASE_DIR.parent / "config" / "schedule.yaml")

    @schedule_app.callback(invoke_without_command=True)
    def schedule_default(
        ctx: typer.Context,
        config: str = typer.Option(default_schedule, "--config", help="Path to schedule YAML."),
    ) -> None:
        if ctx.invoked_subcommand is not None:
            return
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="run"))

    @schedule_app.command("run", help="Run scheduler daemon (time-window services + cron one-shots).")
    def schedule_run(
        config: str = typer.Option(default_schedule, "--config"),
    ) -> None:
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="run"))

    @schedule_app.command("status", help="Show service status (in window, running, pid, log path).")
    def schedule_status(
        config: str = typer.Option(default_schedule, "--config"),
    ) -> None:
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="status"))

    @schedule_app.command("run-now", help="Start a service by id immediately, ignoring its window.")
    def schedule_run_now(
        job_id: str = typer.Argument(..., help="Service id (e.g. weekday_live)."),
        config: str = typer.Option(default_schedule, "--config"),
    ) -> None:
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="run-now", job_id=job_id))

    @schedule_app.command("stop", help="Stop a running service by id.")
    def schedule_stop(
        job_id: str = typer.Argument(..., help="Service id (e.g. weekday_live)."),
        config: str = typer.Option(default_schedule, "--config"),
    ) -> None:
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="stop", job_id=job_id))

    @schedule_app.command("install", help="Install systemd user service so the scheduler runs on startup.")
    def schedule_install(
        config: str = typer.Option(default_schedule, "--config"),
    ) -> None:
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="install"))

    @schedule_app.command("uninstall", help="Remove the systemd user service.")
    def schedule_uninstall(
        config: Optional[str] = typer.Option(default_schedule, "--config"),
    ) -> None:
        from ..run import cmd_schedule

        cmd_schedule(make_args(config=config, schedule_action="uninstall"))
