"""Demo 2 (SnapFit Carrot) parts and simple graspability checks.

Dimensions are the STL bounding boxes from our demo proposal (mm). They are
not fitting tolerances. Colors are the proposal's intent ("contrasting
colors"); update COLOR to match what you actually print, because the
instructions and verifier questions refer to parts by color.

What is NOT known yet (measure on site, then edit GRIPPER below):
* G gripper opening in mm at each command value (0 .. 1.526).
* Fingertip width / length.
* Whether the gripper reading stalls above 0 when it closes on a part.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Part:
    name: str
    size_mm: tuple[float, float, float]  # bounding box x, y, z as printed/lying
    color: str
    note: str = ""

    @property
    def min_side_mm(self) -> float:
        return min(self.size_mm)

    @property
    def grasp_width_mm(self) -> float:
        """Narrowest horizontal side: the width the fingers usually close on."""
        return min(self.size_mm[0], self.size_mm[1])


PARTS: dict[str, Part] = {
    "handle": Part("handle", (70.0, 40.0, 20.0), "white", "snap-fit handle, plate 2"),
    "slider": Part("slider", (60.0, 25.1, 25.1), "orange", "solid toy slider, plate 1"),
    "shell": Part("shell", (70.0, 33.0, 23.5), "green", "outer shell, plate 6"),
    "button_a": Part("button_a", (10.7, 10.7, 12.1), "teal", "snap button, plate 4"),
    "button_b": Part("button_b", (10.7, 10.7, 12.1), "purple", "snap button, plate 4"),
}


@dataclass
class GripperModel:
    """Map gripper command units to finger opening.

    PLACEHOLDER numbers. Replace after measuring on the real G gripper:
    command 0 and 1.526 with calipers, plus two points in between.
    The mapping is likely not linear (revolute fingers).
    """

    max_command: float = 1.526
    max_opening_mm: float = 80.0  # PLACEHOLDER
    fingertip_width_mm: float = 20.0  # PLACEHOLDER

    def opening_mm(self, command: float) -> float:
        command = min(max(command, 0.0), self.max_command)
        return self.max_opening_mm * command / self.max_command

    def command_for_opening(self, opening_mm: float) -> float:
        opening_mm = min(max(opening_mm, 0.0), self.max_opening_mm)
        return self.max_command * opening_mm / self.max_opening_mm


GRIPPER = GripperModel()


def graspability_report(clearance_mm: float = 8.0) -> list[str]:
    """Human-readable check of each part against the gripper model.

    clearance_mm: extra opening needed beyond the part width so the fingers
    can come down around it despite position error (arm repeatability is
    +-1 mm; Wall-X error is much larger).
    """
    lines = []
    for part in PARTS.values():
        need = part.grasp_width_mm + clearance_mm
        ok = need <= GRIPPER.max_opening_mm
        small = part.min_side_mm < GRIPPER.fingertip_width_mm
        status = "OK" if ok else "TOO WIDE"
        extra = " | smaller than fingertip: easy to knock over, hides its own hole" if small else ""
        lines.append(
            f"{part.name:9s} grasp width {part.grasp_width_mm:5.1f} mm, needs opening "
            f"{need:5.1f} mm -> {status}{extra}"
        )
    return lines


if __name__ == "__main__":
    print("Gripper model is a PLACEHOLDER until measured on the robot.\n")
    print("\n".join(graspability_report()))
