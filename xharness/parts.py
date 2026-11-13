"""Demo 2 (SnapFit Carrot) parts and simple graspability checks.

The prop is the team's own four-part design (cover, main body, core, one
integrated pin), not the original MakerWorld toy with two buttons. Source:
the team design repo, `demos/demo_2_snapfit_carrot/design/` (v11 at full
size, v12 at 200%). Sizes are the STL bounding boxes; grip widths and fit
clearances come from that design's `checks/parameters_mm.json` (v11 values,
doubled for v12). Nothing here is measured on a print yet.

PROP_VERSION picks the version the harness and the sim use. v12 is the
newest; its clearances are twice v11's, which matters for the +-1 mm arm.

What is NOT known yet (measure on site, then edit GRIPPER below):
* G gripper opening in mm at each command value (0 .. 1.526).
* Fingertip width / length.
* Whether the gripper reading stalls above 0 when it closes on a part.
* Snap force. A flex snap scaled 2x likely needs roughly 4x the force
  (estimate from beam scaling, unverified): hand-press a print first.
"""

from __future__ import annotations

from dataclasses import dataclass

PROP_VERSION = "v12"  # "v11" (full size) or "v12" (200%)


@dataclass(frozen=True)
class Part:
    name: str
    size_mm: tuple[float, float, float]  # bounding box x, y, z as it lies (the pin: standing, see below)
    color: str
    grip_mm: float  # width the fingers close on, at the part's grip feature
    note: str = ""

    @property
    def min_side_mm(self) -> float:
        return min(self.size_mm)

    @property
    def grasp_width_mm(self) -> float:
        """Width the fingers close on (the grip pad, rib or flat handle, not the whole part)."""
        return self.grip_mm

    @property
    def top_width_mm(self) -> float:
        """Narrowest side seen from above: what a top-down camera measures across the part."""
        return min(self.size_mm[0], self.size_mm[1])


def _v11_parts() -> dict[str, Part]:
    # Sizes from the v11 STLs, reordered so z is the height as the part lies on the table.
    # The pin is listed STANDING (axis vertical, as it goes in): the real pin lies on a sleeve
    # flat for pickup and may need a 90 degree turn in the gripper first. The sim does not model that.
    return {
        "cover": Part("cover", (33.0, 100.0, 26.9), "green", 5.0, "A cover / shell, white grip pad on top"),
        "main": Part("main", (40.0, 100.0, 25.9), "blue", 5.0, "B main body / handle, base of the assembly"),
        "core": Part("core", (27.0, 88.4, 8.6), "orange", 4.0, "C core / slider, picked by the rib between two slots"),
        "pin": Part("pin", (10.0, 10.2, 27.0), "purple", 2.0, "D integrated pin, picked by its flat handle; the snap step"),
    }


def _scaled(parts: dict[str, Part], k: float) -> dict[str, Part]:
    return {n: Part(p.name, tuple(round(s * k, 1) for s in p.size_mm), p.color, p.grip_mm * k, p.note)
            for n, p in parts.items()}


# Radial capture tolerance in mm: how far off-center a part can arrive and still be guided in.
# v11 values from the CAD parameters; v12 doubles them.
#   pin:   (bore 8.0 - nose flat 5.6) / 2 = 1.2  (tapered barbed nose; shaft 7.6 in bore 8.0 is 0.2 per side)
#   core:  (cavity 30.0 - core 27.4) / 2 = 1.3
#   cover: ESTIMATE 3.0, under half the 20 mm diagonal pivot-guide mouth. Rough, not derived exactly.
_V11_FIT_MM = {"pin": 1.2, "core": 1.3, "cover": 3.0}

PARTS_BY_VERSION = {"v11": _v11_parts(), "v12": _scaled(_v11_parts(), 2.0)}
FIT_BY_VERSION = {"v11": dict(_V11_FIT_MM), "v12": {k: v * 2.0 for k, v in _V11_FIT_MM.items()}}

PARTS: dict[str, Part] = PARTS_BY_VERSION[PROP_VERSION]
FIT_MM: dict[str, float] = FIT_BY_VERSION[PROP_VERSION]


@dataclass
class GripperModel:
    """Map gripper command units to finger opening.

    G gripper specs from X Square's Quanta X1 Pro product overview (the mobile robot's
    page; that the desktop kit uses the same G gripper is UNVERIFIED): stroke 0 to 180 mm,
    gripping length 100 mm, max clamping force 50 N, payload 2 kg by friction. Fingertip
    size is not published. The command-to-mm mapping is a PLACEHOLDER: measure command 0
    and 1.526 with calipers, plus two points in between (likely not linear).
    """

    max_command: float = 1.526
    max_opening_mm: float = 180.0  # [doc, mobile robot page] stroke 0 to 180 mm
    max_clamp_force_n: float = 50.0  # [doc, mobile robot page]
    fingertip_width_mm: float = 18.5  # ESTIMATE from the official legacy gripper mesh (team design notes)

    def opening_mm(self, command: float) -> float:
        command = min(max(command, 0.0), self.max_command)
        return self.max_opening_mm * command / self.max_command

    def command_for_opening(self, opening_mm: float) -> float:
        opening_mm = min(max(opening_mm, 0.0), self.max_opening_mm)
        return self.max_command * opening_mm / self.max_opening_mm


GRIPPER = GripperModel()


def graspability_report(clearance_mm: float = 8.0) -> list[str]:
    """Human-readable check of each part's grip feature against the gripper model.

    clearance_mm: extra opening needed beyond the grip width so the fingers
    can come down around it despite position error (arm repeatability is
    +-1 mm; Wall-X error is much larger).
    """
    lines = [f"Prop version {PROP_VERSION}"]
    for part in PARTS.values():
        need = part.grasp_width_mm + clearance_mm
        status = "OK" if need <= GRIPPER.max_opening_mm else "TOO WIDE"
        fit = f", capture +-{FIT_MM[part.name]:.1f} mm" if part.name in FIT_MM else ""
        thin = " | thin grip feature: check fingers can reach it" if part.grasp_width_mm < 5.0 else ""
        lines.append(
            f"{part.name:6s} {part.color:7s} grip {part.grasp_width_mm:5.1f} mm, needs opening "
            f"{need:5.1f} mm -> {status}{fit}{thin}"
        )
    return lines


if __name__ == "__main__":
    print("Gripper model is a PLACEHOLDER until measured on the robot.\n")
    print("\n".join(graspability_report()))
