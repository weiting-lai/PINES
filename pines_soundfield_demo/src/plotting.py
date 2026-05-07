from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


MODEL_LABELS = {
    "pines_tanh": "PINES-Tanh",
    "pines_siren": "PINES-SIREN",
}


def compute_error_percent(true_field: np.ndarray, approx_field: np.ndarray) -> np.ndarray:
    denom = max(float(np.max(np.abs(true_field))) ** 2, float(np.finfo(np.float64).eps))
    return (np.abs(approx_field - true_field) ** 2 / denom) * 100.0


def _smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return data
    return gaussian_filter(data, sigma=float(sigma), mode="nearest")


def plot_soundfield(
    model_name: str,
    true_field: np.ndarray,
    approx_field: np.ndarray,
    n_grid: int,
    region_size: list[float],
    output_path: Path,
    mic_positions: np.ndarray | None,
    field_clim: tuple[float, float] | None = None,
    error_clim: tuple[float, float] = (0.0, 100.0),
    smoothing_sigma: float = 0.85,
) -> np.ndarray:
    true_grid_complex = np.asarray(true_field).reshape(int(n_grid), int(n_grid))
    approx_grid_complex = np.asarray(approx_field).reshape(int(n_grid), int(n_grid))
    true_grid = np.real(true_grid_complex)
    approx_grid = np.real(approx_grid_complex)
    error_percent = compute_error_percent(true_grid_complex, approx_grid_complex)

    if field_clim is None:
        amplitude = float(np.max(np.abs(true_grid)))
        field_clim = (-amplitude, amplitude)

    half_x = float(region_size[0]) / 2.0
    half_y = float(region_size[1]) / 2.0
    extent = [-half_x, half_x, -half_y, half_y]
    panels = [
        ("True Soundfield", _smooth(true_grid, smoothing_sigma), "jet", field_clim, None),
        (f"{MODEL_LABELS.get(model_name, model_name)} Soundfield", _smooth(approx_grid, smoothing_sigma), "jet", field_clim, None),
        ("Error", _smooth(error_percent, smoothing_sigma), "hot_r", error_clim, "[%]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5), facecolor="white")
    mic_xy = None
    if mic_positions is not None:
        mic_arr = np.asarray(mic_positions, dtype=np.float64)
        if mic_arr.ndim == 2 and mic_arr.shape[1] >= 2:
            mic_xy = mic_arr[:, :2]

    for ax, (title, data, cmap, clim, cbar_label) in zip(axes, panels):
        image = ax.imshow(
            data,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            interpolation="bicubic",
            aspect="equal",
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        ax.tick_params(labelsize=11)
        if mic_xy is not None and mic_xy.size:
            ax.scatter(mic_xy[:, 0], mic_xy[:, 1], s=14, c="black", marker="o", linewidths=0.0, zorder=5)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return error_percent
