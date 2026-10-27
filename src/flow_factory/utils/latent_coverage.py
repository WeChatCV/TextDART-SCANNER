# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/utils/latent_coverage.py
"""Rasterize text regions (boxes / quadrilaterals) onto the latent grid.

Coordinates are absolute pixels of an ``image_W x image_H`` image when the size
is given, otherwise a 0-1000 normalized frame. Both rasterizers produce a hard
0/1 mask with edges rounded to the nearest latent cell; a region with positive
area always covers at least one cell.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def _frame_denominators(image_W: Optional[int], image_H: Optional[int]) -> Tuple[float, float]:  # pylint: disable=invalid-name
    if (image_W is None) ^ (image_H is None):
        raise ValueError(
            "image_W and image_H must both be set or both omitted, got "
            f"image_W={image_W!r}, image_H={image_H!r}"
        )
    if image_W is None:
        return 1000.0, 1000.0
    if image_W <= 0 or image_H <= 0:
        raise ValueError(f"expected positive image_W/image_H, got {image_W}x{image_H}")
    return float(image_W), float(image_H)


def quad_aabb_pixel(quad: Sequence[float], image_size: Tuple[int, int]) -> List[int]:
    """Axis-aligned bbox of an 8-number pixel quad, clamped to the image bounds."""
    if len(quad) != 8:
        raise ValueError(f"expected 8-number quad, got len={len(quad)}: {quad!r}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"expected positive image_size=(W,H), got {image_size!r}")
    xs = [float(quad[i]) for i in range(0, 8, 2)]
    ys = [float(quad[i]) for i in range(1, 8, 2)]

    def clamp(value: float, limit: int) -> int:
        return max(0, min(int(round(value)), max(0, limit)))

    return [
        clamp(min(xs), width),
        clamp(min(ys), height),
        clamp(max(xs), width),
        clamp(max(ys), height),
    ]


def box_to_latent_coverage(  # pylint: disable=invalid-name
    box_2d: Sequence[float],
    latent_W: int,
    latent_H: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    *,
    image_W: Optional[int] = None,
    image_H: Optional[int] = None,
) -> torch.Tensor:
    """``(latent_H, latent_W)`` 0/1 coverage of an ``[x0, y0, x1, y1]`` box."""
    denom_w, denom_h = _frame_denominators(image_W, image_H)
    w = torch.zeros((latent_H, latent_W), dtype=dtype, device=device)
    fx0 = float(box_2d[0]) * latent_W / denom_w
    fx1 = float(box_2d[2]) * latent_W / denom_w
    fy0 = float(box_2d[1]) * latent_H / denom_h
    fy1 = float(box_2d[3]) * latent_H / denom_h
    if fx1 < fx0:
        fx0, fx1 = fx1, fx0
    if fy1 < fy0:
        fy0, fy1 = fy1, fy0

    lx0, lx1 = max(0, int(round(fx0))), min(latent_W, int(round(fx1)))
    ly0, ly1 = max(0, int(round(fy0))), min(latent_H, int(round(fy1)))
    if lx1 <= lx0 and fx1 > fx0:
        lx0 = min(lx0, latent_W - 1)
        lx1 = lx0 + 1
    if ly1 <= ly0 and fy1 > fy0:
        ly0 = min(ly0, latent_H - 1)
        ly1 = ly0 + 1
    if lx1 > lx0 and ly1 > ly0:
        w[ly0:ly1, lx0:lx1] = 1.0
    return w


def _point_in_quad(px: float, py: float, xs: Sequence[float], ys: Sequence[float]) -> bool:
    """Ray-casting point-in-polygon test for a closed 4-vertex polygon."""
    inside = False
    j = 3
    for i in range(4):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def quad_to_latent_coverage(  # pylint: disable=invalid-name
    quad_2d: Sequence[float],
    latent_W: int,
    latent_H: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    *,
    image_W: Optional[int] = None,
    image_H: Optional[int] = None,
) -> torch.Tensor:
    """``(latent_H, latent_W)`` 0/1 coverage of an ``[x1, y1, ..., x4, y4]`` quad.

    A cell is covered when its center lies inside the polygon.
    """
    if len(quad_2d) != 8:
        raise ValueError(f"expected 8-number quad_2d, got len={len(quad_2d)}: {quad_2d!r}")
    if latent_W <= 0 or latent_H <= 0:
        raise ValueError(
            f"expected positive latent size, got latent_W={latent_W}, latent_H={latent_H}"
        )
    denom_w, denom_h = _frame_denominators(image_W, image_H)
    xs = [float(quad_2d[i]) * latent_W / denom_w for i in range(0, 8, 2)]
    ys = [float(quad_2d[i]) * latent_H / denom_h for i in range(1, 8, 2)]
    fx0, fx1 = min(xs), max(xs)
    fy0, fy1 = min(ys), max(ys)
    lx0, lx1 = int(round(fx0)), int(round(fx1))
    ly0, ly1 = int(round(fy0)), int(round(fy1))
    lx0, lx1 = max(0, lx0), min(latent_W, max(lx0 + 1, lx1))
    ly0, ly1 = max(0, ly0), min(latent_H, max(ly0 + 1, ly1))

    w = torch.zeros((latent_H, latent_W), dtype=dtype, device=device)
    if lx1 <= lx0 or ly1 <= ly0:
        return w
    for y in range(ly0, ly1):
        for x in range(lx0, lx1):
            if _point_in_quad(x + 0.5, y + 0.5, xs, ys):
                w[y, x] = 1.0

    # Tiny or skinny quads may miss every cell center: paint the cell nearest
    # the AABB center so a non-degenerate region is never dropped.
    if not bool(torch.any(w > 0)) and fx1 > fx0 and fy1 > fy0:
        cx = max(0, min(int(round(0.5 * (fx0 + fx1))), latent_W - 1))
        cy = max(0, min(int(round(0.5 * (fy0 + fy1))), latent_H - 1))
        w[cy, cx] = 1.0
    return w
