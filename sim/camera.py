"""Render the fake world into the three JPEG camera views the protocol needs.

Flat-shaded boxes through a pinhole camera. Good enough for a vision model
to answer "is the purple pin in the gripper?" during offline prompt tests,
and for the protocol to carry realistic-size images. Not photorealistic.

Front camera faces the robot, so (like the real rig) the robot's RIGHT arm
appears on the LEFT of the front image.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from .world import World

COLORS = {
    "white": (235, 235, 230),
    "orange": (240, 140, 30),
    "green": (60, 170, 70),
    "blue": (45, 105, 170),
    "purple": (140, 70, 190),
    "table": (90, 90, 95),
    "gripper": (120, 120, 125),  # light, so fingers are not mistaken for dark holes
    "hole": (12, 12, 20),
}


class PinholeCamera:
    def __init__(self, eye, target, up, width: int, height: int, fov_deg: float):
        self.eye = np.asarray(eye, float)
        forward = np.asarray(target, float) - self.eye
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, np.asarray(up, float))
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.forward)
        self.w, self.h = width, height
        self.f = (width / 2) / np.tan(np.radians(fov_deg) / 2)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixel coords (N,2) and depth (N,)."""
        rel = points - self.eye
        x, y, z = rel @ self.right, rel @ self.up, rel @ self.forward
        z = np.maximum(z, 1e-4)
        u = self.w / 2 + self.f * x / z
        v = self.h / 2 - self.f * y / z
        return np.stack([u, v], axis=1), z


def _box_corners(center_bottom: np.ndarray, size: np.ndarray) -> np.ndarray:
    hx, hy = size[0] / 2, size[1] / 2
    cx, cy, z0 = center_bottom
    z1 = z0 + size[2]
    return np.array([[cx + sx * hx, cy + sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)])


def _hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Convex hull (monotone chain) of 2D points, for filling a projected box."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _render(world: World, cam: PinholeCamera) -> Image.Image:
    img = Image.new("RGB", (cam.w, cam.h), (200, 205, 210))
    draw = ImageDraw.Draw(img)
    # Table
    table = np.array([[0.0, -0.45, 0], [0.75, -0.45, 0], [0.75, 0.45, 0], [0.0, 0.45, 0]])
    uv, _ = cam.project(table)
    draw.polygon([tuple(p) for p in uv], fill=COLORS["table"])

    items = []  # (depth, polygon, color)
    for part in world.parts.values():
        uv, depth = cam.project(_box_corners(part.pos, part.size))
        items.append((float(depth.mean()), _hull(uv), COLORS.get(part.color, (255, 0, 255))))
    for arm in world.arms.values():
        opening = 0.01 + 0.03 * min(arm.gripper, 1.526) / 1.526
        for side in (-1, 1):  # two fingers
            finger = arm.pos + np.array([0.0, side * opening, 0.0])
            uv, depth = cam.project(_box_corners(finger, np.array([0.012, 0.008, 0.05])))
            items.append((float(depth.mean()), _hull(uv), COLORS["gripper"]))
    for _, poly, color in sorted(items, key=lambda it: -it[0]):  # far to near
        if len(poly) >= 3:
            draw.polygon(poly, fill=color, outline=(70, 70, 75))
    # Open holes and slots as dark circles drawn on top, so the finisher can see
    # them even when a held part is above them (a sim simplification: a real part hides its hole).
    for slot in world.slots.values():
        if slot.filled_by or (slot.requires and not world.slots[slot.requires].filled_by):
            continue
        ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        ring = np.stack([slot.xy[0] + slot.radius_m * np.cos(ang), slot.xy[1] + slot.radius_m * np.sin(ang),
                         np.full_like(ang, slot.surface_z)], axis=1)
        uv, _ = cam.project(ring)
        draw.polygon([tuple(p) for p in uv], fill=COLORS["hole"])
    return img


def _jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_views(world: World) -> dict[str, str]:
    """All three camera views as base64 JPEG strings (public_v2 images)."""
    front = PinholeCamera(eye=[0.95, 0.0, 0.55], target=[0.35, 0.0, 0.0], up=[0, 0, 1],
                          width=320, height=240, fov_deg=60)
    views = {"front": _jpeg_b64(_render(world, front))}
    for arm, cam_name in (("left_ee", "left_wrist"), ("right_ee", "right_wrist")):
        tip = world.arms[arm].pos
        # Looks straight down the gripper axis: a held part sits at the image center.
        wrist = PinholeCamera(eye=tip + [0.0, 0, 0.12], target=tip + [0.0, 0, -0.10], up=[1, 0, 0],
                              width=640, height=480, fov_deg=80)
        views[cam_name] = _jpeg_b64(_render(world, wrist))
    return views
