"""Build a transparent, reproducible banner for the GitHub profile.

The galaxy stamps come from the SBSI measurement-flow figure. Probability
contours and the projected large-scale-structure field are generated here.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch
from PIL import Image
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "sbsi-measurement-flow.png"
OUTPUT = ROOT / "assets" / "scientific-ml-banner.png"

BLUE = "#4477AA"
PINK = "#CC6677"
GREEN = "#228833"
AMBER = "#EEA83B"
TEAL = "#2A788E"


def crop_stamps(source: Path) -> list[np.ndarray]:
    """Crop the three simulated galaxy stamps from the SBSI overview figure."""
    image = Image.open(source).convert("RGBA")
    sx = image.width / 2504.0
    sy = image.height / 1779.0
    boxes = [
        (246, 66, 686, 506),
        (1060, 66, 1500, 506),
        (1882, 66, 2322, 506),
    ]
    stamps = []
    for left, top, right, bottom in boxes:
        box = tuple(
            int(round(value * scale))
            for value, scale in zip((left, top, right, bottom), (sx, sy, sx, sy))
        )
        stamps.append(np.asarray(image.crop(box)))
    return stamps


def gaussian_2d(x: np.ndarray, y: np.ndarray, mean, covariance) -> np.ndarray:
    pos = np.stack([x - mean[0], y - mean[1]], axis=-1)
    inverse = np.linalg.inv(np.asarray(covariance))
    exponent = np.einsum("...i,ij,...j->...", pos, inverse, pos)
    return np.exp(-0.5 * exponent)


def simulate_lss(seed: int = 19, grid_size: int = 520) -> np.ndarray:
    """Create a seeded 2D Zel'dovich-style particle-density projection."""
    rng = np.random.default_rng(seed)
    n_modes = 256
    white = rng.normal(size=(n_modes, n_modes))
    white_k = np.fft.fft2(white)

    frequency = np.fft.fftfreq(n_modes)
    kx, ky = np.meshgrid(frequency, frequency, indexing="xy")
    k2 = kx**2 + ky**2
    k = np.sqrt(k2)

    power = (k + 0.012) ** -1.8 * np.exp(-((k / 0.20) ** 4))
    power[0, 0] = 0.0
    delta_k = white_k * np.sqrt(power)
    potential_k = -delta_k / np.where(k2 == 0, 1.0, k2)
    displacement_x = np.fft.ifft2(1j * kx * potential_k).real
    displacement_y = np.fft.ifft2(1j * ky * potential_k).real

    displacement_x /= np.std(displacement_x)
    displacement_y /= np.std(displacement_y)
    q = (np.arange(n_modes) + 0.5) / n_modes
    qx, qy = np.meshgrid(q, q, indexing="xy")
    amplitude = 0.055
    x = (qx + amplitude * displacement_x) % 1.0
    y = (qy + amplitude * displacement_y) % 1.0

    density, _, _ = np.histogram2d(
        y.ravel(), x.ravel(), bins=grid_size, range=((0, 1), (0, 1))
    )
    density = gaussian_filter(density, sigma=2.2)
    density = np.log1p(8.0 * density)
    low, high = np.percentile(density, [35, 99.7])
    return np.clip((density - low) / (high - low), 0.0, 1.0)


def add_connector(fig, start, end, color, bend=0.0, width=1.4, alpha=0.65):
    connector = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        connectionstyle=f"arc3,rad={bend}",
        arrowstyle="-",
        linewidth=width,
        color=color,
        alpha=alpha,
        zorder=2,
    )
    fig.add_artist(connector)


def main() -> None:
    mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})
    fig = plt.figure(figsize=(18, 6), facecolor="none")

    stamps = crop_stamps(SOURCE)
    stamp_colors = [BLUE, PINK, GREEN]
    stamp_positions = [
        (0.018, 0.35, 0.148, 0.44),
        (0.116, 0.21, 0.148, 0.44),
        (0.214, 0.35, 0.148, 0.44),
    ]
    for stamp, color, position in zip(stamps, stamp_colors, stamp_positions):
        ax = fig.add_axes(position, zorder=5)
        ax.imshow(stamp, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.2)

    contour_ax = fig.add_axes((0.405, 0.14, 0.205, 0.72), zorder=4)
    contour_ax.set_xlim(-1.05, 1.05)
    contour_ax.set_ylim(-0.92, 0.92)
    contour_ax.set_aspect("equal")
    contour_ax.axis("off")

    x = np.linspace(-1.05, 1.05, 500)
    y = np.linspace(-0.92, 0.92, 430)
    xx, yy = np.meshgrid(x, y)
    contour_specs = [
        ((-0.23, 0.32), [[0.060, 0.014], [0.014, 0.030]], BLUE),
        ((0.04, -0.17), [[0.078, -0.026], [-0.026, 0.045]], PINK),
        ((0.30, 0.12), [[0.105, 0.016], [0.016, 0.070]], GREEN),
    ]
    for index, (mean, covariance, color) in enumerate(contour_specs):
        density = gaussian_2d(xx, yy, mean, covariance)
        contour_ax.contourf(
            xx,
            yy,
            density,
            levels=[0.14, 0.35, 0.62, 1.01],
            colors=[mpl.colors.to_rgba(color, 0.04),
                    mpl.colors.to_rgba(color, 0.08),
                    mpl.colors.to_rgba(color, 0.14)],
            antialiased=True,
        )
        contour_ax.contour(
            xx,
            yy,
            density,
            levels=[0.14, 0.35, 0.62],
            colors=[color],
            linewidths=[1.1, 1.8, 2.8],
            alpha=0.95,
        )
        contour_ax.scatter(
            mean[0] - 0.035,
            mean[1] + 0.025,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        contour_ax.scatter(
            mean[0] + 0.055,
            mean[1] - 0.035,
            s=95,
            marker="+",
            color=color,
            linewidth=2.2,
            zorder=8,
        )

    lss_ax = fig.add_axes((0.715, 0.075, 0.265, 0.85), zorder=3)
    lss_ax.axis("off")
    density = simulate_lss()
    palette = LinearSegmentedColormap.from_list(
        "lss",
        ["#2A788E", "#55A7A1", "#E4B85A", "#E07B39", "#8A2030"],
    )
    alpha = np.clip((density - 0.08) / 0.78, 0.0, 0.92)
    image = lss_ax.imshow(
        density,
        origin="lower",
        cmap=palette,
        interpolation="bilinear",
        alpha=alpha,
        extent=(0, 1, 0, 1),
    )
    clip = Ellipse((0.50, 0.50), 0.98, 0.88, transform=lss_ax.transData)
    image.set_clip_path(clip)
    lss_contours = lss_ax.contour(
        density,
        levels=[0.28, 0.48, 0.68, 0.84],
        colors=[TEAL, TEAL, AMBER, "#8A2030"],
        linewidths=[0.45, 0.75, 1.0, 1.25],
        alpha=0.58,
        origin="lower",
        extent=(0, 1, 0, 1),
    )
    for collection in lss_contours.collections:
        collection.set_clip_path(clip)
    lss_ax.set_xlim(0, 1)
    lss_ax.set_ylim(0, 1)

    starts = [(0.16, 0.61), (0.26, 0.43), (0.36, 0.61)]
    ends = [(0.44, 0.65), (0.48, 0.40), (0.55, 0.56)]
    for start, end, color, bend in zip(starts, ends, stamp_colors, [-0.08, 0.02, 0.08]):
        add_connector(fig, start, end, color, bend=bend)

    for y0, color, bend in [(0.65, BLUE, -0.09), (0.50, PINK, 0.0), (0.35, GREEN, 0.09)]:
        add_connector(fig, (0.595, y0), (0.735, 0.50), color, bend=bend, width=1.25)

    fig.savefig(OUTPUT, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
