"""Task definitions: a goal broken into stages, loaded from YAML.

One YAML file per A/B-board case. The robot client sends the case's
`instruction` (from the WebUI task JSON) as the goal on every request; the
supervisor picks the task whose `goal` matches it.

Example (see config/tasks/ for real ones):

    task_id: d2_case3_pin
    goal: "Insert the purple pin into the toy until it snaps in."
    context: "Toy assembly at the center; the pin lies on the table to one side."
    stages:
      - id: grasp_pin
        active_arm: right_ee
        instructions:
          - "Pick up the purple pin with the right arm."
          - "Grasp the purple pin by its white handle."
        done:
          question: "Is the purple pin held between the right gripper fingers, off the table?"
          confirmations: 2
          gripper_holding: right_ee
        budget_requests: 12
        max_retries: 2

Field meanings are documented on the dataclasses below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .protocol import ARMS


@dataclass
class DoneCheck:
    # Question for the vision model. Phrase it so "yes" means the stage is done.
    question: str
    # Consecutive "yes" answers needed (debounce against one lucky frame).
    confirmations: int = 2
    # Also require the arms to be still (end-effector moved < threshold).
    require_still: bool = False
    # Also require the gripper reading of this arm to be in the "holding" band.
    gripper_holding: str | None = None
    # Also require the gripper of this arm to be open (part released).
    gripper_open: str | None = None


@dataclass
class Stage:
    id: str
    # Instruction phrasings for Wall-X. Retries rotate to the next phrasing,
    # because VLA behavior can change a lot with wording.
    instructions: list[str]
    done: DoneCheck
    # The arm doing the work (used by freeze_idle_arm). None = both / unknown.
    active_arm: str | None = None
    # Wall-X requests allowed per attempt before we count a retry.
    budget_requests: int = 15
    # Attempts after the first one. Exhausted -> task failed (or skipped).
    max_retries: int = 2
    # If true and retries are exhausted, move on instead of failing the task.
    optional: bool = False
    # A pure checking stage: the relay answers HOLD (arms stay still, Wall-X is
    # not called) while the vision model is asked the done question on every
    # request. Used to look at the result once the gripper is out of the way.
    hold: bool = False
    # "wallx" (default): Wall-X moves the arms. "finisher": xharness/finisher.py moves
    # the active arm (final alignment, press, release, lift) and decides when it is done.
    type: str = "wallx"
    part: str = ""  # finisher only: key in xharness/parts.py of the held part
    target: str = ""  # finisher only: name of the hole or slot (documentation and logs)
    # finisher only: gripper z in metres (robot base frame, as in observation.state) when the part is
    # fully seated. MEASURE ON SITE. None = the finisher aborts before pressing (never press blind).
    seated_z: float | None = None
    # Which mode this stage belongs to: None = both, "finisher" = only when
    # finisher.enabled, "wallx" = only when it is not.
    only: str | None = None
    # On failure (retries exhausted, or a finisher abort), jump back to this stage
    # instead of failing the task, at most max_loops times. Honored only in finisher mode.
    on_fail_goto: str | None = None
    max_loops: int = 2


@dataclass
class Task:
    task_id: str
    goal: str
    stages: list[Stage]
    context: str = ""  # scene description passed to the vision model
    match: str = ""  # optional substring match on the goal instead of exact match

    def matches(self, goal: str) -> bool:
        goal_n = _norm(goal)
        if self.match:
            return _norm(self.match) in goal_n
        return goal_n == _norm(self.goal)


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().rstrip(".").split())


def load_task(path: str | Path, finisher_enabled: bool = False) -> Task:
    """Load one task YAML. Stages marked `only` are kept for their mode only."""
    data = yaml.safe_load(Path(path).read_text())
    stages = []
    for raw in data["stages"]:
        only = raw.get("only")
        if only not in (None, "finisher", "wallx"):
            raise ValueError(f"{path}: stage {raw.get('id')} has only={only!r}")
        if only is not None and (only == "finisher") != finisher_enabled:
            continue
        done = DoneCheck(**raw.pop("done", {"question": "(decided by the finisher controller)"}))
        raw.setdefault("instructions", [])  # hold stages have none
        stage = Stage(done=done, **raw)
        if not stage.instructions and not stage.hold and stage.type != "finisher":
            raise ValueError(f"{path}: stage {stage.id} has no instructions")
        if stage.type not in ("wallx", "finisher"):
            raise ValueError(f"{path}: stage {stage.id} type must be wallx or finisher")
        if stage.type == "finisher":
            if not stage.part:
                raise ValueError(f"{path}: finisher stage {stage.id} needs a part")
            stage.instructions = stage.instructions or []
        if stage.active_arm not in (None, *ARMS):
            raise ValueError(f"{path}: stage {stage.id} active_arm must be left_ee or right_ee")
        stages.append(stage)
    task = Task(
        task_id=data["task_id"],
        goal=data["goal"],
        stages=stages,
        context=data.get("context", ""),
        match=data.get("match", ""),
    )
    ids = [s.id for s in task.stages]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate stage ids")
    for s_ in task.stages:
        if s_.on_fail_goto and s_.on_fail_goto not in ids[: ids.index(s_.id)]:
            if finisher_enabled:  # in wallx mode the target stage may be filtered out
                raise ValueError(f"{path}: {s_.id} on_fail_goto must name an EARLIER stage")
    return task


def load_tasks(paths: list[str], finisher_enabled: bool = False) -> list[Task]:
    return [load_task(p, finisher_enabled) for p in paths]
