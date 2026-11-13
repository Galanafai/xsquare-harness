"""Command line entry point.

    python -m xharness --config config/harness.example.yaml --host 192.168.1.20
    python -m xharness --check             # validate config + tasks, print warnings, exit
    python -m xharness --reset-state       # forget task progress (between practice runs)
    python -m xharness --no-prewarm        # skip the startup warm-up request to Wall-X
    python -m xharness --hold-only         # reply HOLD to everything (safe on-site HOLD test)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import load_config
from .parts import graspability_report
from .relay import Relay
from .supervisor import Supervisor
from .tasks import load_tasks
from .verifier import Verifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xharness", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/harness.example.yaml")
    parser.add_argument("--task", action="append", default=[], help="task YAML (repeatable; adds to config task_files)")
    parser.add_argument("--host", help="LAN IP the robot can reach (overrides config)")
    parser.add_argument("--port", type=int, help="port (overrides config)")
    parser.add_argument("--passthrough", action="store_true", help="ignore tasks: forward the client's goal unchanged")
    parser.add_argument("--hold-only", action="store_true",
                        help="answer every request with HOLD (never calls Wall-X). Use once on site to "
                             "confirm the robot accepts our HOLD chunks and stays still.")
    parser.add_argument("--no-prewarm", action="store_true",
                        help="skip the synthetic warm-up request sent to Wall-X at startup (never sent with --hold-only)")
    parser.add_argument("--reset-state", action="store_true", help="forget saved task progress before starting")
    parser.add_argument("--check", action="store_true", help="validate config and tasks, print warnings, then exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.host:
        cfg.listen.host = args.host
    if args.port:
        cfg.listen.port = args.port
    task_files = [] if (args.passthrough or args.hold_only) else [*cfg.task_files, *args.task]
    if args.hold_only:
        cfg.supervisor.unknown_goal = "hold"
    tasks = load_tasks(task_files, cfg.finisher.enabled)
    verifier = Verifier(cfg.verifier)
    supervisor = Supervisor(tasks, cfg.supervisor, verifier)
    if args.reset_state:
        supervisor.reset()

    warnings = preflight_warnings(cfg, tasks, verifier)
    print("xharness preflight")
    print(f"  Wall-X endpoint : {cfg.wallx.endpoint}")
    print(f"  auth headers    : {'file set' if cfg.wallx.headers_file else 'NONE'}")
    print(f"  listen          : ws://{cfg.listen.host}:{cfg.listen.port}")
    print(f"  tasks           : {[t.task_id for t in tasks] or 'none (passthrough)'}")
    print(f"  verifier        : {'on, ' + cfg.verifier.model if verifier.available else 'OFF (stages can never complete)'}")
    print(f"  filter mode     : {cfg.filter.mode}   hold on done: {cfg.hold.on_done}")
    for w in warnings:
        print(f"  WARNING: {w}")
    if args.check:
        print("\nPart check (gripper numbers are placeholders until measured):")
        print("  " + "\n  ".join(graspability_report()))
        return 0

    try:
        asyncio.run(Relay(cfg, supervisor, prewarm=not (args.no_prewarm or args.hold_only)).serve_forever())
    except KeyboardInterrupt:
        print("\nharness stopped. Confirm the robot session is stopped in the WebUI.")
    return 0


def preflight_warnings(cfg, tasks, verifier) -> list[str]:
    out = []
    if cfg.listen.host in ("127.0.0.1", "localhost"):
        out.append("listening on localhost: the robot cannot reach this. Use your LAN IP.")
    if not cfg.wallx.headers_file and cfg.wallx.endpoint.startswith("wss://"):
        out.append("no WALLX_HEADERS_FILE: the cloud endpoint will likely reject the handshake.")
    if tasks and not verifier.available:
        out.append("tasks loaded but verifier is off: no stage can ever be marked done.")
    if cfg.filter.mode == "enforce" or cfg.filter.freeze_idle_arm or cfg.hold.on_done:
        out.append("harness may REPLACE Wall-X actions (HOLD / filter). Confirm organizers allow this.")
    default_box = all(b.min == [-1.0, -1.0, -1.0] and b.max == [1.0, 1.0, 1.0] for b in cfg.filter.workspace.values())
    if default_box:
        out.append("workspace box is the placeholder default. Measure it on site from recorded states.")
    return out


if __name__ == "__main__":
    sys.exit(main())
