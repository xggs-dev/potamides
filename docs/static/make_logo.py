# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
"""Draw the potamides logo: a capital Phi, its ring a stellar stream.

The name's "P" is Phi, the gravitational potential. Here its ring is a stream
of stars, scattered from a fixed seed and shaded teal into purple along it, with
arrows pointing in from the stream: its curvature, matched to the potential's
pull. The stem is a short bar with serifs, so the letter is a capital. The
shapes are vector, so the logo is written as an SVG, sharp at any size; for a
bitmap, name a .png and give its size::

    uv run docs/static/make_logo.py                     # favicon.svg
    uv run docs/static/make_logo.py --size 2048 big.png
"""

import argparse
import math
import random
from pathlib import Path

NAVY, TEAL, PURPLE, CYAN = "#030a23", "#66a19a", "#7738eb", "#9ff8ff"

# In a 64-unit square. The ring: its centre's x and y, half-width, half-height.
RING = (32, 32, 16, 13.5)
STARS, SEED, SCATTER = 120, 1, 0.06  # the stream's stars, their draw and spread
STAR_SIZE = (0.9, 1.9)  # the stars' smallest and largest radius
STEM = (10, 54, 6)  # the stem's top and bottom y, and its serifs' half-width
# The curvature arrows: where on the ring (degrees, clockwise from the right),
# three a side, and how far each reaches in.
ARROWS, REACH = (325, 0, 35, 145, 180, 215), 6
HEAD = (2.0, 1.5)  # an arrowhead's length, and its half-width

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="512" height="512">
  <rect width="64" height="64" rx="14" fill="{navy}"/>
{stars}
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="{stem}" stroke="{cyan}" stroke-width="3.6"/>
    <path d="{arrows}" stroke="#fff" stroke-width="1.6"/>
  </g>
</svg>
"""


def blend(t: float) -> str:
    """Return the colour ``t`` of the way from teal to purple."""
    a, b = (bytes.fromhex(c[1:]) for c in (TEAL, PURPLE))
    return "#" + "".join(
        f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b, strict=False)
    )


def stars() -> str:
    """Return the stream: stars scattered about the ring, as SVG circles."""
    cx, cy, rx, ry = RING
    rng = random.Random(SEED)
    circles = []
    for i in range(STARS):
        t = i / STARS
        angle = 2 * math.pi * t - math.pi / 2  # from the top, clockwise
        r = 1 + rng.gauss(0, SCATTER)
        x, y = cx + rx * r * math.cos(angle), cy + ry * r * math.sin(angle)
        size = rng.uniform(*STAR_SIZE)
        circles.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{blend(t)}"/>'
        )
    return "\n".join(circles)


def arrows() -> str:
    """Return the curvature arrows, each a shaft and a head, as one SVG path."""
    cx, cy, rx, ry = RING
    paths = []
    for degrees in ARROWS:
        a = math.radians(degrees)
        x0, y0 = cx + rx * math.cos(a), cy + ry * math.sin(a)
        x1 = cx + (rx - REACH) * math.cos(a)
        y1 = cy + (ry - 0.77 * REACH) * math.sin(a)
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length  # along the shaft
        px, py = -uy, ux  # across it
        back_x, back_y = x1 - HEAD[0] * ux, y1 - HEAD[0] * uy
        side_x, side_y = HEAD[1] * px, HEAD[1] * py
        paths.append(
            f"M{x0:.2f} {y0:.2f}L{x1:.2f} {y1:.2f}"
            f"M{back_x + side_x:.2f} {back_y + side_y:.2f}L{x1:.2f} {y1:.2f}"
            f"L{back_x - side_x:.2f} {back_y - side_y:.2f}"
        )
    return "".join(paths)


def svg() -> str:
    """Return the logo as SVG text."""
    cx = RING[0]
    top, bottom, serif = STEM
    stem = f"M{cx:g} {top:g}V{bottom:g}"
    stem += f"M{cx - serif:g} {top:g}H{cx + serif:g}"
    stem += f"M{cx - serif:g} {bottom:g}H{cx + serif:g}"
    return SVG.format(navy=NAVY, cyan=CYAN, stars=stars(), stem=stem, arrows=arrows())


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
