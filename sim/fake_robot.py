"""A stand-in for the organizers' robot client (x2robot_client), in sync mode.

It does what the real client does, as documented:
  connect -> read handshake -> loop {observe, send request, receive reply,
  execute rows 0..30} -> on disconnect, reconnect (the guide says the real
  client "may reconnect", which is why closing the harness is not a stop).

Observations come from sim/world.py, images from sim/camera.py, so the
harness under test sees exactly the public_v2 shapes it will see on Saturday.

Fault injection on the client side:
  drop_connection_at: request numbers after which the client disconnects
                      and reconnects (tests that progress survives).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from xharness.protocol import (
    CONTROL_CONTEXT_PARAMS,
    MAX_FRAME_BYTES,
    is_error_response,
    pack,
    unpack,
    validate_response,
)

from .camera import render_views
from .world import STAGE_TRUTH, World


@dataclass
class ClientReport:
    requests: int = 0
    executed_chunks: int = 0
    hold_like_chunks: int = 0
    error_replies: int = 0
    rejected_chunks: int = 0
    reconnects: int = 0
    elapsed_s: float = 0.0
    world_events: list[str] = field(default_factory=list)
    truth: dict[str, bool] = field(default_factory=dict)  # at the end of the run
    ever_true: set[str] = field(default_factory=set)  # achieved at some point
    # (wall time, stages true right then), one entry per executed chunk
    timeline: list[tuple[float, frozenset]] = field(default_factory=list)


class FakeRobotClient:
    def __init__(self, world: World, url: str, goal: str, embodiment: str = "quanta_x1_pro/gripper-g",
                 max_requests: int = 120, time_budget_s: float = 480.0, max_reconnects: int = 5,
                 drop_connection_at: tuple[int, ...] = (), verbose: bool = False):
        self.world = world
        self.url = url
        self.goal = goal
        self.embodiment = embodiment
        self.max_requests = max_requests
        self.time_budget_s = time_budget_s  # the A/B board allows 8 minutes per case
        self.max_reconnects = max_reconnects
        self.drop_at = set(drop_connection_at)
        self.verbose = verbose

    def observation(self) -> dict:
        return {
            "embodiment": self.embodiment,
            "normalization": "provided",
            "instruction": self.goal,
            "observation": {"images": render_views(self.world), "state": self.world.state()},
            "control_context": {arm: dict.fromkeys(CONTROL_CONTEXT_PARAMS) for arm in ("left", "right")},  # client extension field, shape per live handshake
        }

    async def run(self) -> ClientReport:
        report = ClientReport()
        started = time.monotonic()
        while report.requests < self.max_requests and report.reconnects <= self.max_reconnects:
            try:
                async with connect(self.url, max_size=MAX_FRAME_BYTES, ping_interval=None,
                                   open_timeout=10, compression=None) as ws:
                    unpack(await ws.recv())  # handshake
                    while report.requests < self.max_requests:
                        if time.monotonic() - started > self.time_budget_s:
                            break
                        await ws.send(pack(self.observation()))
                        report.requests += 1
                        reply = unpack(await ws.recv())
                        self._apply(reply, report)
                        if report.requests in self.drop_at:
                            self.drop_at.discard(report.requests)
                            break  # leave the context: disconnect, then reconnect
                    else:
                        break
                if time.monotonic() - started > self.time_budget_s:
                    break
                if report.requests < self.max_requests:
                    report.reconnects += 1
            except (ConnectionClosed, OSError) as err:
                report.reconnects += 1
                if self.verbose:
                    print(f"[fake robot] connection lost ({type(err).__name__}); reconnecting")
                await asyncio.sleep(0.05)
        report.elapsed_s = round(time.monotonic() - started, 2)
        report.world_events = list(self.world.events)
        report.truth = {stage: bool(fn(self.world)) for stage, fn in STAGE_TRUTH.items()}
        return report

    def _apply(self, reply, report: ClientReport) -> None:
        if is_error_response(reply):
            report.error_replies += 1
            return
        try:
            actions = validate_response(reply, self.embodiment)
        except Exception:  # noqa: BLE001 - a real client would refuse to execute this
            report.rejected_chunks += 1
            return
        if all((actions[a]["position"] == actions[a]["position"][0]).all() for a in actions):
            report.hold_like_chunks += 1
        self.world.execute(actions)
        report.executed_chunks += 1
        now_true = frozenset(stage for stage, fn in STAGE_TRUTH.items() if fn(self.world))
        report.ever_true |= now_true
        report.timeline.append((time.time(), now_true))
