"""A stand-in for the Wall-X service, for offline harness testing.

It speaks the same public_v2 WebSocket protocol, but instead of a neural
network it uses a scripted "oracle" policy that cheats by reading the fake
world. Position noise (resampled every chunk) makes grasps and insertions
fail sometimes, roughly like a zero-shot VLA, so retries get exercised.

Fault injection lets you rehearse every failure the real service can show:

    FaultPlan({3: "delay:8", 5: "error:missing_field", 7: "close1011",
               9: "nan", 11: "jump", 13: "text_frame", 15: "bad_shape"})

Keys are 1-based request numbers counted across the whole server lifetime.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from xharness.parts import GRIPPER
from xharness.protocol import ACTION_ROWS, ARMS, ProtocolError, build_response, pack, unpack, validate_control_context, validate_request

from .world import World

COLOR_TO_PART = {"orange": "slider", "green": "shell", "teal": "button_a", "purple": "button_b", "white": "handle"}
MAX_TRAVEL_PER_CHUNK = 0.15  # metres; keeps the fake arm speed plausible


@dataclass
class FaultPlan:
    at_request: dict[int, str] = field(default_factory=dict)

    @classmethod
    def standard(cls) -> "FaultPlan":
        """A rehearsal of every failure type, spread through one run."""
        return cls({4: "delay:3", 7: "error:missing_field", 10: "close1011", 13: "nan",
                    16: "jump", 19: "text_frame", 22: "bad_shape"})


class FakeWallX:
    def __init__(self, world: World, noise_m: float = 0.003, warmup_s: float = 0.0,
                 latency_s: float = 0.0, faults: FaultPlan | None = None, seed: int = 0):
        self.world = world
        self.noise_m = noise_m
        self.warmup_s = warmup_s
        self.latency_s = latency_s
        self.faults = faults or FaultPlan()
        self.rng = np.random.default_rng(seed)
        self.requests = 0
        self.instructions: list[str] = []

    async def handler(self, ws: ServerConnection) -> None:
        await ws.send(pack({"model": "wall-x", "model_version": "fake-oracle"}))
        first = True
        try:
            async for frame in ws:
                self.requests += 1
                n = self.requests
                fault = self.faults.at_request.get(n, "")
                if first and self.warmup_s:
                    await asyncio.sleep(self.warmup_s)
                first = False
                if self.latency_s:
                    await asyncio.sleep(self.latency_s)
                try:
                    request = unpack(frame)
                    validate_request(request)
                except ProtocolError as err:
                    await ws.send(pack({"error": {"code": "missing_field", "message": str(err)}}))
                    continue
                try:
                    validate_control_context(request)
                except ProtocolError as err:
                    await ws.send(pack({"error": {"code": "control_context_invalid", "message": str(err)}}))
                    continue
                self.instructions.append(request["instruction"])

                if fault.startswith("delay:"):
                    await asyncio.sleep(float(fault.split(":")[1]))
                elif fault.startswith("error:"):
                    await ws.send(pack({"error": {"code": fault.split(":")[1], "message": "injected"}}))
                    continue
                elif fault == "close1011":
                    await ws.close(code=1011, reason="injected server error")
                    return
                elif fault == "text_frame":
                    await ws.send("Traceback (most recent call last): injected")
                    await ws.close(code=1011, reason="injected")
                    return

                actions = self.plan(request)
                if fault == "nan":
                    actions["left_ee"]["position"][5, 0] = np.nan
                elif fault == "jump":
                    actions["right_ee"]["position"][0] += [0.0, 0.0, 0.4]
                elif fault == "bad_shape":
                    actions["left_ee"]["gripper"] = actions["left_ee"]["gripper"][:16]
                await ws.send(pack(build_response(request["embodiment"], actions)))
        except ConnectionClosed:
            pass

    async def serve(self, host: str = "127.0.0.1", port: int = 0):
        return await serve(self.handler, host, port, ping_interval=None, compression=None)

    # ------------------------------------------------------------ policy

    def plan(self, request: dict) -> dict:
        """Turn an instruction into a 32-row chunk for both arms."""
        text = request["instruction"].lower()
        state = request["observation"]["state"]
        current = {a: np.asarray(state[a]["position"], float) for a in ARMS}
        rpy = {a: np.asarray(state[a]["rotation_rpy"], float) for a in ARMS}
        grip = {a: float(state[a]["gripper"]) for a in ARMS}

        part = next((p for c, p in COLOR_TO_PART.items() if c in text), None)
        holder = next((a for a in ARMS if self.world.arms[a].holding), None)
        arm = holder or self._nearest_arm(part, current)

        goal = current[arm].copy()
        goal_grip = grip[arm]
        noise = np.append(self.rng.normal(0, self.noise_m, 2), 0.0)
        verb = _verb(text)

        if verb == "grasp" and holder and self.world.arms[holder].holding == part:
            goal = current[arm] + [0.0, 0.0, 0.06]  # already holding it: lift
            goal_grip = 0.0
        elif verb == "grasp" and part in self.world.parts and not self.world.arms[arm].holding:
            target = self.world.parts[part]
            grasp_z = target.pos[2] + target.size[2] / 2
            above = np.array([target.pos[0], target.pos[1], grasp_z + 0.06]) + noise
            if np.linalg.norm(current[arm][:2] - above[:2]) > 0.01 or current[arm][2] > grasp_z + 0.08:
                goal, goal_grip = above, GRIPPER.max_command  # approach open
            else:
                goal = np.array([current[arm][0], current[arm][1], grasp_z])
                goal_grip = 0.0  # close; the world stalls it on the part
        elif verb in ("align", "insert") and holder:
            held = self.world.arms[holder].holding
            slot = self.world.free_slot_for(held)
            if slot is not None:
                size_z = self.world.parts[held].size[2]
                hover = slot.seat_z + size_z / 2 + (0.03 if verb == "align" else -0.004)
                goal = np.array([slot.xy[0], slot.xy[1], hover]) + noise
                goal_grip = 0.0
        elif verb == "release":
            arm = min(ARMS, key=lambda a: grip[a])  # the arm that is closed
            goal = current[arm] + [0.0, 0.0, 0.08]
            goal_grip = GRIPPER.max_command

        return self._trajectory(arm, current, rpy, grip, goal, goal_grip)

    def _nearest_arm(self, part: str | None, current: dict) -> str:
        if part in self.world.parts:
            xy = self.world.parts[part].pos[:2]
            return min(ARMS, key=lambda a: np.linalg.norm(current[a][:2] - xy))
        return "right_ee"

    def _trajectory(self, arm, current, rpy, grip, goal, goal_grip) -> dict:
        actions = {}
        for a in ARMS:
            start = current[a]
            end = goal if a == arm else start  # idle arm stays put
            step = end - start
            dist = np.linalg.norm(step)
            if dist > MAX_TRAVEL_PER_CHUNK:
                end = start + step / dist * MAX_TRAVEL_PER_CHUNK
            # Move during rows 0..19, then hold position; close the gripper
            # only after arriving (rows 22+). Opening happens immediately.
            t = np.clip(np.arange(ACTION_ROWS) / 19.0, 0.0, 1.0)[:, None]
            pos = start + t * (end - start)
            pos[-1] = pos[-2]  # row 31 repeats row 30, like the real service
            g_end = goal_grip if a == arm else grip[a]
            closing = g_end < grip[a]
            g = np.where(np.arange(ACTION_ROWS) < 22, grip[a], g_end) if closing else np.full(ACTION_ROWS, g_end)
            actions[a] = {"position": pos, "rotation_rpy": np.tile(rpy[a], (ACTION_ROWS, 1)), "gripper": g.astype(float)}
        return actions


def _verb(text: str) -> str:
    if re.search(r"\b(release|let go|move .*arms? up|away)\b", text):
        return "release"
    if re.search(r"\b(insert|press|push|snap|seat)\b", text):
        return "insert"
    if re.search(r"\b(above|align|line up|hold the .* (just )?above)\b", text):
        return "align"
    if re.search(r"\b(pick|grasp|grab|lift)\b", text):
        return "grasp"
    return "none"
