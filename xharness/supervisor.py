"""The supervisor: decides, for every robot request, what happens next.

For each observation the robot client sends, the supervisor returns one of:

* ACT   -> forward to Wall-X with this sub-instruction.
* HOLD  -> do not call Wall-X; send a chunk that keeps the arms still.
* PASS  -> forward the client's goal unchanged (no task matched this goal).

It owns the plan (a Task = list of Stages), progress, retries and the
"are we done" decision. Wall-X owns *how* to move; the supervisor owns
*what stage we are in*.

Why a fixed stage list instead of a free-form LLM planner?
* The competition is fully autonomous: nobody may touch a keyboard once a
  run starts, so stopping and recovery must be predictable.
* A wrong free-form plan is hard to debug in an 8-minute case; a stage
  table is readable and testable offline.

Progress survives reconnects. The example harness kept progress per
WebSocket connection, so a client reconnect silently restarted the task
from stage 1 while the arm was mid-motion. Here progress is keyed by the
goal text and saved to disk after every change.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import SupervisorConfig
from .protocol import ARMS, ArmState
from .tasks import Stage, Task
from .verifier import Verdict, Verifier

ACT, HOLD, PASS = "act", "hold", "pass"


@dataclass
class Decision:
    kind: str  # act | hold | pass
    instruction: str = ""
    stage_id: str = ""
    active_arm: str | None = None
    reason: str = ""
    events: list[dict] = field(default_factory=list)  # for the recorder


@dataclass
class RunState:
    """Progress through one task. Serialized to JSON."""

    task_id: str
    goal: str
    stage_index: int = 0
    attempt: int = 0  # 0 = first try of the current stage
    requests_in_attempt: int = 0
    consecutive_yes: int = 0
    status: str = "running"  # running | done | failed
    started_at: float = field(default_factory=time.time)
    stage_log: list[dict] = field(default_factory=list)


class Supervisor:
    def __init__(self, tasks: list[Task], cfg: SupervisorConfig, verifier: Verifier):
        self.tasks = tasks
        self.cfg = cfg
        self.verifier = verifier
        self.state_path = Path(cfg.state_file)
        self.runs: dict[str, RunState] = self._load()
        # Not persisted: only used to measure motion between two requests.
        self._last_positions: dict[str, dict[str, np.ndarray]] = {}
        self._last_command_gripper: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ API

    def reset(self, goal: str | None = None) -> None:
        """Forget progress for one goal (or all). Use between practice runs."""
        if goal is None:
            self.runs.clear()
        else:
            self.runs.pop(goal, None)
        self._save()

    def note_commanded_gripper(self, goal: str, last_rows: dict[str, float]) -> None:
        """Remember the final commanded gripper value of the chunk we forwarded."""
        self._last_command_gripper[goal] = dict(last_rows)

    async def decide(self, request: dict, state: dict[str, ArmState]) -> Decision:
        goal = request["instruction"].strip()
        run = self._run_for(goal)
        if run is None:
            kind = PASS if self.cfg.unknown_goal == "passthrough" else HOLD
            return Decision(kind, instruction=goal, reason="no task matches this goal")

        task = self._task(run.task_id)
        still = self._update_motion(goal, state)

        if run.status == "done":
            return Decision(HOLD, reason="task complete")
        if run.status == "failed":
            return Decision(HOLD, reason="task failed: retries exhausted")

        stage = task.stages[run.stage_index]
        run.requests_in_attempt += 1
        events: list[dict] = []

        # 1) Is the current stage done?
        if self._should_verify(run):
            verdict, gripper_ok, still_ok = await self._check_done(task, stage, request, state, goal, still)
            events.append(
                {
                    "event": "verify",
                    "stage": stage.id,
                    "answer": verdict.answer,
                    "reason": verdict.reason,
                    "error": verdict.error,
                    "latency_ms": verdict.latency_ms,
                    "gripper_ok": gripper_ok,
                    "still_ok": still_ok,
                }
            )
            if verdict.yes and gripper_ok and still_ok:
                run.consecutive_yes += 1
            elif verdict.answer == "no" or not gripper_ok:
                run.consecutive_yes = 0
            # "unsure" neither counts nor resets: a blocked view is not evidence.

            if run.consecutive_yes >= stage.done.confirmations:
                events.append({"event": "stage_done", "stage": stage.id})
                if not self._advance(run, task, "done"):
                    self._save()
                    return Decision(HOLD, reason="task complete", events=events)
                stage = task.stages[run.stage_index]
                run.requests_in_attempt = 1

        # 2) Out of budget for this attempt? Retry with the next phrasing.
        if run.requests_in_attempt > stage.budget_requests:
            run.attempt += 1
            run.requests_in_attempt = 1
            run.consecutive_yes = 0
            events.append({"event": "retry", "stage": stage.id, "attempt": run.attempt})
            if run.attempt > stage.max_retries:
                if stage.optional:
                    events.append({"event": "stage_skipped", "stage": stage.id})
                    if not self._advance(run, task, "skipped"):
                        self._save()
                        return Decision(HOLD, reason="task complete (last stage skipped)", events=events)
                    stage = task.stages[run.stage_index]
                    run.requests_in_attempt = 1
                else:
                    run.status = "failed"
                    run.stage_log.append({"stage": stage.id, "result": "failed", "t": time.time()})
                    events.append({"event": "task_failed", "stage": stage.id})
                    self._save()
                    return Decision(HOLD, stage_id=stage.id, reason="retries exhausted", events=events)

        # 3) Act: send the current stage's instruction to Wall-X.
        instruction = stage.instructions[run.attempt % len(stage.instructions)]
        self._save()
        return Decision(
            ACT,
            instruction=instruction,
            stage_id=stage.id,
            active_arm=stage.active_arm,
            reason=f"stage {run.stage_index + 1}/{len(task.stages)}, attempt {run.attempt + 1}",
            events=events,
        )

    # ------------------------------------------------------------ internals

    def _run_for(self, goal: str) -> RunState | None:
        if goal in self.runs:
            return self.runs[goal]
        for task in self.tasks:
            if task.matches(goal):
                self.runs[goal] = RunState(task_id=task.task_id, goal=goal)
                self._save()
                return self.runs[goal]
        return None

    def _task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def _should_verify(self, run: RunState) -> bool:
        return run.requests_in_attempt % max(1, self.cfg.verify_every) == 0

    def _advance(self, run: RunState, task: Task, result: str) -> bool:
        """Move to the next stage. Returns False when the task is finished."""
        run.stage_log.append({"stage": task.stages[run.stage_index].id, "result": result, "t": time.time()})
        run.stage_index += 1
        run.attempt = 0
        run.consecutive_yes = 0
        if run.stage_index >= len(task.stages):
            run.status = "done"
            return False
        return True

    async def _check_done(
        self, task: Task, stage: Stage, request: dict, state: dict[str, ArmState], goal: str, still: bool
    ) -> tuple[Verdict, bool, bool]:
        gripper_ok = self._gripper_condition(stage, state, goal)
        still_ok = still or not stage.done.require_still
        if not gripper_ok:
            # Cheap check failed: skip the slow vision call.
            return Verdict("no", "gripper condition not met", 0.0), False, still_ok
        verdict = await self.verifier.check(f"{task.task_id}.{stage.id}", stage.done.question, request, task.context)
        return verdict, gripper_ok, still_ok

    def _gripper_condition(self, stage: Stage, state: dict[str, ArmState], goal: str) -> bool:
        lo, hi = self.cfg.grasp_band
        arm = stage.done.gripper_holding
        if arm:
            g = state[arm].gripper
            if not (lo <= g <= hi):
                return False
            commanded = self._last_command_gripper.get(goal, {}).get(arm)
            if commanded is not None and g < commanded + self.cfg.stall_margin:
                # Fingers reached the commanded closure: nothing blocked them.
                return False
        arm = stage.done.gripper_open
        if arm and state[arm].gripper < self.cfg.open_above:
            return False
        return True

    def _update_motion(self, goal: str, state: dict[str, ArmState]) -> bool:
        current = {arm: state[arm].position for arm in ARMS}
        previous = self._last_positions.get(goal)
        self._last_positions[goal] = current
        if previous is None:
            return False
        moved = max(float(np.linalg.norm(current[a] - previous[a])) for a in ARMS)
        return moved < self.cfg.still_threshold_m

    def _load(self) -> dict[str, RunState]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text())
            return {goal: RunState(**data) for goal, data in raw.items()}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({g: asdict(r) for g, r in self.runs.items()}, indent=2))
        tmp.replace(self.state_path)
