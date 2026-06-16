#!/usr/bin/env python3
"""Generate a printable OpenCV chessboard calibration pattern on A4 paper.

The pattern is rendered in landscape A4 orientation, with the cols dimension
(horizontal) being the longer axis — matching the standard OpenCV convention
where pattern_size = (cols, rows) with cols > rows.

Usage:
    python gen_chessboard.py                      # default: 11x8, auto-sized ~80% A4
    python gen_chessboard.py --cols 9 --rows 6    # custom pattern size
    python gen_chessboard.py --square-size 20     # force square size in mm

Output: chessboard_11x8.pdf (vector PDF, print at 100% / "Actual Size")
"""

import argparse
import math

from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Landscape A4 in mm
PAGE_W, PAGE_H = 297.0, 210.0


def compute_square_size(cols: int, rows: int, area_fraction: float = 0.80) -> float:
    """Compute square size (mm) so pattern fills ~area_fraction of A4.

    Pattern has (cols+1) squares horizontally and (rows+1) vertically,
    placed on a landscape A4 page.
    """
    sq_w = cols + 1
    sq_h = rows + 1

    target_area = area_fraction * PAGE_W * PAGE_H
    s = math.sqrt(target_area / (sq_w * sq_h))

    min_margin = 8.0
    s = min(s, (PAGE_W - 2 * min_margin) / sq_w,
               (PAGE_H - 2 * min_margin) / sq_h)

    return max(math.floor(s * 2) / 2, 5.0)


def draw_chessboard(cols: int, rows: int, square_size: float,
                    output: str = "chessboard_11x8.pdf") -> None:
    """Draw chessboard pattern centered on landscape A4 and save as PDF."""
    sq_w = cols + 1
    sq_h = rows + 1
    pat_w = sq_w * square_size
    pat_h = sq_h * square_size

    ox = (PAGE_W - pat_w) / 2.0
    oy = (PAGE_H - pat_h) / 2.0

    # Figure sized as landscape A4
    fig = Figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=72)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw black squares. Top-left square (r=0, c=0) is white, so black
    # squares are where (r + c) is odd.
    for r in range(sq_h):
        for c in range(sq_w):
            if (r + c) % 2 == 0:
                continue
            x = ox + c * square_size
            y = oy + r * square_size
            rect = Rectangle((x, y), square_size, square_size,
                              facecolor="black", edgecolor="black", linewidth=0)
            ax.add_patch(rect)

    # Thin border
    border = Rectangle((ox, oy), pat_w, pat_h,
                        facecolor="none", edgecolor="black", linewidth=0.5)
    ax.add_patch(border)

    # Annotation at bottom center
    info = (f"Pattern: {cols}×{rows} inner corners  |  "
            f"Squares: {sq_w}×{sq_h}  |  "
            f"Cell: {square_size:.1f} mm  |  "
            f"Size: {pat_w:.1f}×{pat_h:.1f} mm")
    ax.text(PAGE_W / 2, 3, info, ha="center", va="bottom",
            fontsize=5, color="gray")

    canvas = FigureCanvasPdf(fig)
    canvas.print_figure(output)

    area_pct = pat_w * pat_h / (PAGE_W * PAGE_H) * 100
    print(f"Saved {output}")
    print(f"  Pattern : {cols}x{rows} inner corners ({sq_w}x{sq_h} squares)")
    print(f"  Cell    : {square_size:.1f} mm")
    print(f"  Size    : {pat_w:.1f} x {pat_h:.1f} mm")
    print(f"  Area    : {area_pct:.1f}% of A4")
    print(f"  Margins : {(PAGE_W - pat_w) / 2:.1f} mm sides, "
          f"{(PAGE_H - pat_h) / 2:.1f} mm top/bottom")
    print(f"\nOpenCV usage:")
    print(f'  ret, corners = cv2.findChessboardCorners(img, ({cols}, {rows}))')
    print(f'  objp = ... * {square_size:.1f}  # mm')


def main():
    parser = argparse.ArgumentParser(
        description="Generate printable chessboard calibration pattern")
    parser.add_argument("--cols", type=int, default=11,
                        help="Inner corners in X (default: 11)")
    parser.add_argument("--rows", type=int, default=8,
                        help="Inner corners in Y (default: 8)")
    parser.add_argument("--square-size", type=float, default=None,
                        help="Force square size in mm (default: auto ~80%% A4)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (default: chessboard_{cols}x{rows}.pdf)")
    args = parser.parse_args()

    sq = args.square_size or compute_square_size(args.cols, args.rows)
    out = args.output or f"chessboard_{args.cols}x{args.rows}.pdf"
    draw_chessboard(args.cols, args.rows, sq, out)


if __name__ == "__main__":
    main()
