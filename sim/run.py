"""Run one Demo 2 case end to end on a laptop, no robot needed.

    python -m sim.run --case 3
    python -m sim.run --case 3 --faults standard --vlm-faults messy
    python -m sim.run --case 1 --drop-connection 6 --noise 0.004
    python -m sim.run --case 3 --real-wallx       # real cloud Wall-X, fake robot + fake vision

Pieces started in this one process:

    fake robot client --> xharness relay --> fake Wall-X (or the real one)
                                  \\--> fake vision model (HTTP)

At the end it prints what the supervisor BELIEVED happened next to what
ACTUALLY happened in the world. Disagreements are the bugs to look for:
a stage the supervisor marked done that the world says is not done is a
false positive, the most dangerous failure in a no-touch run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from xharness.config import HarnessConfig, load_config
from xharness.relay import Relay
from xharness.supervisor import Supervisor
from xharness.tasks import load_task
from xharness.verifier import Verifier

from .fake_robot import FakeRobotClient
from .fake_vlm import FakeVLM
from .fake_wallx import FakeWallX, FaultPlan
from .world import World

CASE_TASKS = {
    1: "config/tasks/d2_case1_insert_slider.yaml",
    2: "config/tasks/d2_case2_snap_shell.yaml",
    3: "config/tasks/d2_case3_press_buttons.yaml",
}
VLM_FAULTS = {
    "none": {},
    "messy": {"fenced": 0.3, "prose": 0.2, "http500": 0.05, "wrong": 0.03},
    "hostile": {"fenced": 0.3, "prose": 0.3, "http500": 0.15, "slow": 0.1, "wrong": 0.1},
}


async def run_case(case: int, *, noise: float = 0.003, wallx_faults: FaultPlan | None = None,
                   vlm_faults: dict | None = None, drop_connection: tuple[int, ...] = (),
                   real_wallx: bool = False, max_requests: int = 150, seed: int = 0,
                   place_offset_mm: tuple[float, float] | None = None,
                   cfg: HarnessConfig | None = None, verbose: bool = True) -> dict:
    world = World(case, seed=seed)
    cfg = cfg or load_config(None)
    task = load_task(CASE_TASKS[case], cfg.finisher.enabled)
    workdir = Path(tempfile.mkdtemp(prefix="xharness-sim-"))

    for stage in task.stages:  # on site a person measures this by hand; in sim we read it from the world
        if stage.type == "finisher":
            stage.seated_z = world.seated_gripper_z(stage.part)
    vlm = FakeVLM(world, faults=vlm_faults or {}, slow_s=3.0, seed=seed)
    vlm_url = vlm.start()

    cfg.listen.host, cfg.listen.port = "127.0.0.1", 0
    cfg.verifier.chat_url, cfg.verifier.model, cfg.verifier.timeout_s = vlm_url, "fake-oracle", 2.0
    cfg.supervisor.state_file = str(workdir / "supervisor_state.json")
    cfg.recorder.dir = str(workdir / "runs")
    cfg.recorder.frame_every = 5

    fake_wallx = None
    wallx_server = None
    if real_wallx:
        cfg.wallx.endpoint = os.environ.get("WALLX_ENDPOINT", cfg.wallx.endpoint)
        cfg.wallx.headers_file = os.environ.get("WALLX_HEADERS_FILE", cfg.wallx.headers_file)
    else:
        fake_wallx = FakeWallX(world, noise_m=noise, faults=wallx_faults, seed=seed, place_offset_mm=place_offset_mm)
        wallx_server = await fake_wallx.serve()
        port = wallx_server.sockets[0].getsockname()[1]
        cfg.wallx.endpoint = f"ws://127.0.0.1:{port}"
        cfg.wallx.headers_file = ""
        cfg.wallx.first_request_timeout_s = cfg.wallx.request_timeout_s = 2.0

    supervisor = Supervisor([task], cfg.supervisor, Verifier(cfg.verifier), cfg.finisher, cfg.filter)
    relay = Relay(cfg, supervisor, verbose=verbose)
    from websockets.asyncio.server import serve

    harness_server = await serve(relay.handle, "127.0.0.1", 0, ping_interval=None, compression=None,
                                 max_size=32 * 1024 * 1024)
    harness_url = f"ws://127.0.0.1:{harness_server.sockets[0].getsockname()[1]}"

    client = FakeRobotClient(world, harness_url, task.goal, max_requests=max_requests,
                             drop_connection_at=drop_connection, verbose=verbose)
    try:
        report = await client.run()
    finally:
        harness_server.close()
        await harness_server.wait_closed()
        if wallx_server:
            wallx_server.close()
            await wallx_server.wait_closed()
        vlm.stop()

    run = supervisor.runs.get(task.goal)
    premature = _premature_advances(Path(cfg.recorder.dir), report.timeline)
    believed = {entry["stage"]: entry["result"] for entry in (run.stage_log if run else [])}
    comparison = []
    for stage in task.stages:
        ever = stage.id in report.ever_true
        belief = believed.get(stage.id, "not reached")
        flag = ""
        if belief == "done" and stage.id in premature:
            flag = "FALSE POSITIVE"  # supervisor advanced while the world said not done
        elif belief in ("failed", "not reached") and ever:
            flag = "missed (happened, not detected)"
        comparison.append({"stage": stage.id, "supervisor": belief, "world_ever": ever, "flag": flag})

    return {
        "case": case,
        "task_status": run.status if run else "no run",
        "requests": report.requests,
        "executed_chunks": report.executed_chunks,
        "hold_chunks": report.hold_like_chunks,
        "error_replies": report.error_replies,
        "rejected_chunks": report.rejected_chunks,
        "reconnects": report.reconnects,
        "vlm_calls": vlm.calls,
        "wallx_requests": fake_wallx.requests if fake_wallx else "real",
        "stages": comparison,
        "world_events": report.world_events[-12:],
        "run_dir": str(workdir),
    }


def _premature_advances(runs_dir: Path, timeline: list) -> set[str]:
    """Stages the supervisor marked done at a moment the world said they were not."""
    premature = set()
    for events_file in runs_dir.glob("*/events.jsonl"):
        for line in events_file.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("event") != "stage_done":
                continue
            before = [truth for t, truth in timeline if t <= ev["t"]]
            if before and ev["stage"] not in before[-1]:
                premature.add(ev["stage"])
    return premature


def print_summary(result: dict) -> None:
    print("\n" + "=" * 72)
    print(f"Case {result['case']}: task {result['task_status'].upper()}  "
          f"({result['requests']} requests, {result['executed_chunks']} chunks executed, "
          f"{result['hold_chunks']} HOLD, {result['reconnects']} reconnects)")
    print(f"errors forwarded: {result['error_replies']}   chunks a client would reject: {result['rejected_chunks']}"
          f"   vision calls: {result['vlm_calls']}")
    print("-" * 72)
    print(f"{'stage':18s} {'supervisor says':16s} {'world: happened?':17s}")
    for row in result["stages"]:
        print(f"{row['stage']:18s} {row['supervisor']:16s} {str(row['world_ever']):17s} {row['flag']}")
    print("-" * 72)
    print("last world events:")
    for e in result["world_events"]:
        print("  " + e)
    print(f"\nlogs and frames: {result['run_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--noise", type=float, default=0.003, help="fake Wall-X xy position noise (m)")
    parser.add_argument("--faults", choices=("none", "standard"), default="none", help="Wall-X fault plan")
    parser.add_argument("--vlm-faults", choices=tuple(VLM_FAULTS), default="none")
    parser.add_argument("--drop-connection", type=int, action="append", default=[],
                        help="client disconnects after this request number (repeatable)")
    parser.add_argument("--real-wallx", action="store_true", help="use the real endpoint (needs WALLX_HEADERS_FILE)")
    parser.add_argument("--max-requests", type=int, default=150)
    parser.add_argument("--finisher", action="store_true", help="enable the finisher (final alignment and press)")
    parser.add_argument("--place-offset-mm", type=float, nargs=2, metavar=("LO", "HI"),
                        help="fake Wall-X leaves every align this far off (mm range)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    cfg = load_config(None)
    cfg.finisher.enabled = args.finisher
    result = asyncio.run(run_case(
        args.case, cfg=cfg, place_offset_mm=tuple(args.place_offset_mm) if args.place_offset_mm else None, noise=args.noise,
        wallx_faults=FaultPlan.standard() if args.faults == "standard" else None,
        vlm_faults=VLM_FAULTS[args.vlm_faults], drop_connection=tuple(args.drop_connection),
        real_wallx=args.real_wallx, max_requests=args.max_requests, seed=args.seed, verbose=not args.quiet,
    ))
    print(json.dumps(result, indent=2) if args.json else "")
    if not args.json:
        print_summary(result)


if __name__ == "__main__":
    main()
