from __future__ import annotations

# This file keeps the compact metrics/noise helpers used by the demo. Portions
# mirror MIT-licensed helper code from:
# - PIDL-sound-field-reconstruction, https://github.com/steDamiano/PIDL-sound-field-reconstruction
# - local_soundfield_reconstruction, https://github.com/manvhah/local_soundfield_reconstruction

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.spatial
import torch
import yaml
from scipy.special import hankel1


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_name


def add_complex_awgn_measured(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    signal = np.asarray(x, dtype=np.complex128).reshape(-1)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    if signal_power <= 0.0 or not np.isfinite(signal_power):
        return signal
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    scale = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)) * scale
    return signal + noise


def nmse_db_and_ncc(estimated: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    est = torch.as_tensor(np.asarray(estimated).reshape(-1), dtype=torch.complex64)
    gt = torch.as_tensor(np.asarray(ground_truth).reshape(-1), dtype=torch.complex64)
    nmse = torch.sum(torch.abs(est - gt) ** 2) / (torch.sum(torch.abs(gt) ** 2) + 1e-12)
    ncc = torch.abs(est @ torch.conj(gt)) / ((torch.norm(est, p=2) * torch.norm(gt, p=2)) + 1e-12)
    return float(10.0 * torch.log10(nmse + 1e-12).item()), float(ncc.item())


def frequency_axis(sample_rate: int, fft_size: int) -> np.ndarray:
    return np.linspace(0.0, float(sample_rate) / 2.0, int(fft_size) // 2 + 1)


def sample_sources_outside_roi(
    seed: int,
    n_sources: int,
    room_size: list[float],
    region_size: list[float],
    region_origin: list[float],
    margin: float,
    max_trials: int = 100_000,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    room = np.asarray(room_size, dtype=np.float64).reshape(-1)
    roi = np.asarray(region_size, dtype=np.float64).reshape(-1)
    origin = np.asarray(region_origin, dtype=np.float64).reshape(-1)
    if room.shape != roi.shape or room.shape != origin.shape:
        raise ValueError("room_size, region_size, and region_origin must have matching dimensions")

    lower = origin - float(margin)
    upper = origin + roi + float(margin)
    out = np.zeros((int(n_sources), room.size), dtype=np.float64)
    for idx in range(int(n_sources)):
        for _ in range(int(max_trials)):
            candidate = rng.uniform(float(margin), room - float(margin))
            if not bool(np.all((lower <= candidate) & (candidate <= upper))):
                out[idx, :] = candidate
                break
        else:
            raise RuntimeError("Failed to sample source positions outside the ROI")
    return out


def centered_to_absolute(centered: np.ndarray, region_size: list[float], region_origin: list[float]) -> np.ndarray:
    coords = np.asarray(centered, dtype=np.float64)
    region = np.asarray(region_size, dtype=np.float64).reshape(-1)
    origin = np.asarray(region_origin, dtype=np.float64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != region.size:
        raise ValueError(f"centered coordinates must have shape (N, {region.size}), got {coords.shape}")
    return coords + (origin + region / 2.0).reshape(1, -1)


def generate_centered_lattice(region_size: list[float], grid_count: int) -> np.ndarray:
    region = np.asarray(region_size, dtype=np.float64).reshape(-1)
    axes = [np.linspace(-side / 2.0, side / 2.0, int(grid_count), dtype=np.float64) for side in region]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([axis.ravel() for axis in mesh]).astype(np.float64)


def generate_centered_xy_slice(region_size: list[float], n_grid: int, slice_z: float) -> np.ndarray:
    region = np.asarray(region_size, dtype=np.float64).reshape(3)
    xs = np.linspace(-region[0] / 2.0, region[0] / 2.0, int(n_grid), dtype=np.float64)
    ys = np.linspace(-region[1] / 2.0, region[1] / 2.0, int(n_grid), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, float(slice_z), dtype=np.float64)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)


def freefield_response(
    receiver_positions: np.ndarray,
    source_positions: np.ndarray,
    sample_rate: int,
    rir_length: int,
    fft_size: int,
    sound_speed: float,
) -> np.ndarray:
    receivers = np.asarray(receiver_positions, dtype=np.float64)
    sources = np.asarray(source_positions, dtype=np.float64)
    if receivers.ndim != 2 or sources.ndim != 2 or receivers.shape[1] != sources.shape[1]:
        raise ValueError(f"receiver/source dimensions must match, got {receivers.shape} and {sources.shape}")

    h = np.zeros((receivers.shape[0], int(rir_length)), dtype=np.float64)
    fs = float(sample_rate)
    c = float(sound_speed)
    for recv_idx, recv in enumerate(receivers):
        for source in sources:
            dist = max(float(np.linalg.norm(recv - source)), 1e-10)
            delay_idx = int(round((dist / c) * fs))
            if 0 <= delay_idx < int(rir_length):
                h[recv_idx, delay_idx] += 1.0 / (4.0 * np.pi * dist)
        norm = np.linalg.norm(h[recv_idx])
        if norm > 0.0:
            h[recv_idx, :] /= norm

    spectra = np.zeros((receivers.shape[0], int(fft_size)), dtype=np.complex64)
    for idx in range(receivers.shape[0]):
        spectra[idx, :] = np.fft.fft(h[idx, :], n=int(fft_size))
    return spectra


def room_response(
    receiver_positions: np.ndarray,
    source_positions: np.ndarray,
    room_size: list[float],
    sample_rate: int,
    rir_length: int,
    fft_size: int,
    sound_speed: float,
    t60: float,
    ism_order: int,
    randomized_ism: bool,
) -> np.ndarray:
    if float(t60) <= 0.0:
        return freefield_response(receiver_positions, source_positions, sample_rate, rir_length, fft_size, sound_speed)

    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise ImportError("pyroomacoustics is required when t60 > 0. Install requirements.txt first.") from exc

    receivers = np.asarray(receiver_positions, dtype=np.float64)
    sources = np.asarray(source_positions, dtype=np.float64)
    absorption, sabine_order = pra.inverse_sabine(float(t60), np.asarray(room_size, dtype=np.float64))
    room = pra.ShoeBox(
        np.asarray(room_size, dtype=np.float64),
        fs=int(sample_rate),
        materials=pra.Material(absorption),
        max_order=int(ism_order or sabine_order),
        use_rand_ism=bool(randomized_ism),
        max_rand_disp=0.08,
    )
    for source in sources:
        room.add_source(source)
    room.add_microphone_array(receivers.T)
    room.compute_rir()

    h_all = np.zeros((receivers.shape[0], int(rir_length)), dtype=np.float64)
    for mic_idx in range(receivers.shape[0]):
        for src_idx in range(sources.shape[0]):
            rir = np.asarray(room.rir[mic_idx][src_idx], dtype=np.float64)
            length = min(rir.shape[0], int(rir_length))
            if length > 0:
                h_all[mic_idx, :length] += rir[:length]
        norm = np.linalg.norm(h_all[mic_idx])
        if norm > 0.0:
            h_all[mic_idx, :] /= norm

    spectra = np.zeros((receivers.shape[0], int(fft_size)), dtype=np.complex64)
    for idx in range(receivers.shape[0]):
        spectra[idx, :] = np.fft.fft(h_all[idx, :], n=int(fft_size))
    return spectra


def compute_hankel_2d_matrix(receiver_xy: np.ndarray, source_xy: np.ndarray, k: float) -> np.ndarray:
    dist = scipy.spatial.distance.cdist(np.asarray(receiver_xy, dtype=np.float64), np.asarray(source_xy, dtype=np.float64))
    dist = np.maximum(dist, 1e-10)
    return (-1j / 4.0 * hankel1(0, float(k) * dist)).astype(np.complex128)


def compute_green_3d_matrix(receiver_xyz: np.ndarray, source_xyz: np.ndarray, k: float) -> np.ndarray:
    receiver = np.asarray(receiver_xyz, dtype=np.float64)
    source = np.asarray(source_xyz, dtype=np.float64)
    diff = receiver[:, None, :] - source[None, :, :]
    radius = np.sqrt(np.maximum(np.sum(diff**2, axis=2), 1e-20))
    return (np.exp(1j * float(k) * radius) / (4.0 * np.pi * radius)).astype(np.complex128)


def build_dual_ring_geometry(cfg: dict[str, Any]) -> dict[str, Any]:
    anchors = int(cfg.get("anchors", 80))
    radii = [float(value) for value in cfg.get("radii", [1.55, 1.70])]
    phase_stagger = bool(cfg.get("phase_stagger", True))
    use_ds_weighting = bool(cfg.get("use_ds_weighting", True))
    if anchors <= 0 or not radii:
        raise ValueError("2D equivalent source geometry requires positive anchors and at least one radius")

    base = anchors // len(radii)
    remainder = anchors % len(radii)
    ring_counts = [base + (1 if idx < remainder else 0) for idx in range(len(radii))]
    points: list[np.ndarray] = []
    ds_values: list[float] = []
    for idx, (radius, count) in enumerate(zip(radii, ring_counts)):
        if count <= 0:
            continue
        phase = (np.pi / count) * idx if phase_stagger else 0.0
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + phase
        points.append(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]))
        ds_values.extend([(2.0 * np.pi * radius) / count] * count)

    ds = np.asarray(ds_values, dtype=np.float32)
    if not use_ds_weighting:
        ds[:] = 1.0
    return {
        "point_cloud": np.vstack(points).astype(np.float32),
        "ds": ds,
        "anchors": int(ds.shape[0]),
        "radii": radii,
        "ring_counts": ring_counts,
        "geometry": "dual-ring",
        "use_ds_weighting": use_ds_weighting,
    }


def fibonacci_sphere_points(count: int, radius: float) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    indices = np.arange(count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    xy_radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    theta = golden_angle * indices
    unit = np.column_stack([np.cos(theta) * xy_radius, np.sin(theta) * xy_radius, z])
    return (float(radius) * unit).astype(np.float64)


def build_dual_fibonacci_sphere_geometry(cfg: dict[str, Any]) -> dict[str, Any]:
    anchors = int(cfg.get("anchors", 96))
    radii = [float(value) for value in cfg.get("radii", [1.1, 1.25])]
    use_ds_weighting = bool(cfg.get("use_ds_weighting", True))
    if anchors <= 0 or not radii:
        raise ValueError("3D equivalent source geometry requires positive anchors and at least one radius")

    base = anchors // len(radii)
    remainder = anchors % len(radii)
    shell_counts = [base + (1 if idx < remainder else 0) for idx in range(len(radii))]
    points: list[np.ndarray] = []
    ds_parts: list[np.ndarray] = []
    for radius, count in zip(radii, shell_counts):
        if count <= 0:
            continue
        points.append(fibonacci_sphere_points(count, radius))
        ds_parts.append(np.full(count, 4.0 * np.pi * radius * radius / count, dtype=np.float32))

    ds = np.concatenate(ds_parts).astype(np.float32)
    if not use_ds_weighting:
        ds[:] = 1.0
    return {
        "point_cloud": np.vstack(points).astype(np.float32),
        "ds": ds,
        "anchors": int(ds.shape[0]),
        "radii": radii,
        "shell_counts": shell_counts,
        "geometry": "dual-fibonacci-sphere",
        "use_ds_weighting": use_ds_weighting,
    }
