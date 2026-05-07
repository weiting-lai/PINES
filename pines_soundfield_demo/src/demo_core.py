from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.pines import build_pines_model, count_trainable_params
from src.utils import (
    add_complex_awgn_measured,
    build_dual_fibonacci_sphere_geometry,
    build_dual_ring_geometry,
    centered_to_absolute,
    compute_green_3d_matrix,
    compute_hankel_2d_matrix,
    frequency_axis,
    generate_centered_lattice,
    generate_centered_xy_slice,
    nmse_db_and_ncc,
    resolve_device,
    room_response,
    sample_sources_outside_roi,
    write_json,
)


ALLOWED_MODELS = ("pines_tanh", "pines_siren")


def run_root_name(
    seed: int,
    fold_idx: int,
    bin_index: int,
    mics: int,
    anchors: int,
    case_tag: str | None,
    extra: str = "",
) -> str:
    base = f"seed{int(seed)}_fold{int(fold_idx)}_bin{int(bin_index)}_m{int(mics)}_a{int(anchors)}"
    if extra:
        base = f"{base}_{extra}"
    if case_tag:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_tag.strip())
        if safe:
            base = f"{base}_{safe}"
    return base


def make_2d_case(case_cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(case_cfg.get("seed", 42))
    fold_idx = int(case_cfg.get("fold", 0))
    fold_seed = seed + fold_idx
    room_cfg = dict(case_cfg["room"])
    n_grid = int(case_cfg.get("n_grid", 24))
    mics = int(case_cfg.get("mics", 16))
    bin_index = int(case_cfg.get("frequency_bin", 40))
    sample_rate = int(room_cfg.get("sample_rate", 16000))
    fft_size = int(room_cfg.get("fft_size", 1024))
    freqs = frequency_axis(sample_rate, fft_size)
    if not 0 <= bin_index < len(freqs):
        raise ValueError(f"frequency_bin={bin_index} outside valid range [0, {len(freqs) - 1}]")

    sources_abs = sample_sources_outside_roi(
        seed=fold_seed,
        n_sources=int(room_cfg.get("n_sources", 2)),
        room_size=room_cfg["room_size"],
        region_size=room_cfg["region_size"],
        region_origin=room_cfg["region_origin"],
        margin=float(room_cfg.get("source_margin", 0.1)),
    )
    grid_centered = generate_centered_lattice(room_cfg["region_size"], n_grid)
    if mics > grid_centered.shape[0]:
        raise ValueError(f"mics={mics} exceeds available grid points={grid_centered.shape[0]}")
    rng = np.random.default_rng(fold_seed)
    mic_indices = np.sort(rng.choice(grid_centered.shape[0], size=mics, replace=False))
    mic_centered = grid_centered[mic_indices]

    grid_abs = centered_to_absolute(grid_centered, room_cfg["region_size"], room_cfg["region_origin"])
    spectra = room_response(
        receiver_positions=grid_abs,
        source_positions=sources_abs,
        room_size=room_cfg["room_size"],
        sample_rate=sample_rate,
        rir_length=int(room_cfg.get("rir_length", 2048)),
        fft_size=fft_size,
        sound_speed=float(room_cfg.get("sound_speed", 343.0)),
        t60=float(room_cfg.get("t60", 0.0)),
        ism_order=int(room_cfg.get("ism_order", 10)),
        randomized_ism=bool(room_cfg.get("randomized_ism", False)),
    )

    return {
        "dimension": 2,
        "fold_seed": fold_seed,
        "room_cfg": room_cfg,
        "freq_axis": freqs,
        "bin_index": bin_index,
        "frequency_hz": float(freqs[bin_index]),
        "mics": mics,
        "n_grid": n_grid,
        "source_positions": sources_abs,
        "mic_coords": mic_centered,
        "eval_coords": grid_centered,
        "mic_pressure": spectra[mic_indices, bin_index],
        "ground_truth": spectra[:, bin_index],
    }


def make_3d_case(case_cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(case_cfg.get("seed", 42))
    fold_idx = int(case_cfg.get("fold", 0))
    fold_seed = seed + fold_idx
    room_cfg = dict(case_cfg["room"])
    n_grid = int(case_cfg.get("n_grid", 22))
    mic_grid = int(case_cfg.get("mic_grid", 5))
    mics = int(case_cfg.get("mics", 24))
    slice_z = float(case_cfg.get("slice_z", 0.0))
    bin_index = int(case_cfg.get("frequency_bin", 40))
    sample_rate = int(room_cfg.get("sample_rate", 16000))
    fft_size = int(room_cfg.get("fft_size", 1024))
    freqs = frequency_axis(sample_rate, fft_size)
    if not 0 <= bin_index < len(freqs):
        raise ValueError(f"frequency_bin={bin_index} outside valid range [0, {len(freqs) - 1}]")

    sources_abs = sample_sources_outside_roi(
        seed=fold_seed,
        n_sources=int(room_cfg.get("n_sources", 2)),
        room_size=room_cfg["room_size"],
        region_size=room_cfg["region_size"],
        region_origin=room_cfg["region_origin"],
        margin=float(room_cfg.get("source_margin", 0.1)),
    )
    mic_candidates = generate_centered_lattice(room_cfg["region_size"], mic_grid)
    if mics > mic_candidates.shape[0]:
        raise ValueError(f"mics={mics} exceeds available 3D mic candidates={mic_candidates.shape[0]}")
    rng = np.random.default_rng(fold_seed)
    mic_indices = np.sort(rng.choice(mic_candidates.shape[0], size=mics, replace=False))
    mic_centered = mic_candidates[mic_indices]
    slice_centered = generate_centered_xy_slice(room_cfg["region_size"], n_grid, slice_z)

    receiver_centered = np.vstack([mic_centered, slice_centered])
    receiver_abs = centered_to_absolute(receiver_centered, room_cfg["region_size"], room_cfg["region_origin"])
    spectra = room_response(
        receiver_positions=receiver_abs,
        source_positions=sources_abs,
        room_size=room_cfg["room_size"],
        sample_rate=sample_rate,
        rir_length=int(room_cfg.get("rir_length", 2048)),
        fft_size=fft_size,
        sound_speed=float(room_cfg.get("sound_speed", 343.0)),
        t60=float(room_cfg.get("t60", 0.0)),
        ism_order=int(room_cfg.get("ism_order", 10)),
        randomized_ism=bool(room_cfg.get("randomized_ism", False)),
    )

    return {
        "dimension": 3,
        "fold_seed": fold_seed,
        "room_cfg": room_cfg,
        "freq_axis": freqs,
        "bin_index": bin_index,
        "frequency_hz": float(freqs[bin_index]),
        "mics": mics,
        "n_grid": n_grid,
        "slice_z": slice_z,
        "source_positions": sources_abs,
        "mic_coords": mic_centered,
        "eval_coords": slice_centered,
        "mic_pressure": spectra[:mics, bin_index],
        "ground_truth": spectra[mics:, bin_index],
    }


def _variant_defaults(model_name: str) -> dict[str, Any]:
    if model_name == "pines_tanh":
        return {
            "family": "baseline_tanh_shared",
            "hidden_dim": 64,
            "hidden_layers": 2,
            "pe_levels": 0,
            "coord_scale": 2.0,
            "lr": 1e-3,
            "reg_coeff": 1e-5,
            "iterations": 300,
        }
    if model_name == "pines_siren":
        return {
            "family": "siren_shared",
            "hidden_dim": 64,
            "hidden_layers": 2,
            "coord_scale": 2.0,
            "w0": 30.0,
            "lr": 1e-3,
            "reg_coeff": 1e-6,
            "iterations": 300,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _model_cfg(pines_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    cfg = _variant_defaults(model_name)
    cfg.update({key: value for key, value in pines_cfg.get("models", {}).get(model_name, {}).items() if value is not None})
    return cfg


def _geometry_for_case(pines_cfg: dict[str, Any], dimension: int) -> dict[str, Any]:
    if dimension == 2:
        return build_dual_ring_geometry(dict(pines_cfg.get("equivalent_sources_2d", {})))
    if dimension == 3:
        return build_dual_fibonacci_sphere_geometry(dict(pines_cfg.get("equivalent_sources_3d", {})))
    raise ValueError(f"Unsupported dimension: {dimension}")


def _transfer_matrix(dimension: int, receivers: np.ndarray, sources: np.ndarray, k: float) -> np.ndarray:
    if dimension == 2:
        return compute_hankel_2d_matrix(receivers, sources, k)
    if dimension == 3:
        return compute_green_3d_matrix(receivers, sources, k)
    raise ValueError(f"Unsupported dimension: {dimension}")


def train_pines_model(
    model_name: str,
    case_data: dict[str, Any],
    pines_cfg: dict[str, Any],
    output_dir: Path,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    metrics_path = output_dir / "metrics.json"
    estimated_path = output_dir / "pressure_estimated.npy"
    gt_path = output_dir / "pressure_gt.npy"
    if resume and metrics_path.exists() and estimated_path.exists() and gt_path.exists():
        import json

        return json.loads(metrics_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "status.json", {"status": "running", "model_name": model_name})

    dimension = int(case_data["dimension"])
    model_cfg = _model_cfg(pines_cfg, model_name)
    geometry = _geometry_for_case(pines_cfg, dimension)
    device = resolve_device(device_name)
    seed = int(case_data["fold_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_points = np.asarray(geometry["point_cloud"], dtype=np.float64)
    ds = torch.tensor(geometry["ds"], dtype=torch.float32, device=device)
    arch_cfg = {
        key: model_cfg[key]
        for key in ("family", "hidden_dim", "hidden_layers", "pe_levels", "coord_scale", "w0")
        if key in model_cfg
    }
    model = build_pines_model(arch_cfg, source_points).to(device)
    param_count = count_trainable_params(model)

    freq_hz = float(case_data["frequency_hz"])
    sound_speed = float(case_data["room_cfg"].get("sound_speed", 343.0))
    k = 2.0 * np.pi * freq_hz / sound_speed
    mic_pressure = add_complex_awgn_measured(
        case_data["mic_pressure"],
        snr_db=float(pines_cfg.get("mic_snr_db", 30.0)),
        seed=seed + 1_000_003 * int(case_data["bin_index"]),
    )
    ground_truth = np.asarray(case_data["ground_truth"], dtype=np.complex128).reshape(-1)

    train_matrix = torch.tensor(
        _transfer_matrix(dimension, case_data["mic_coords"], source_points, k),
        dtype=torch.complex64,
        device=device,
    )
    eval_matrix = torch.tensor(
        _transfer_matrix(dimension, case_data["eval_coords"], source_points, k),
        dtype=torch.complex64,
        device=device,
    )
    target = torch.tensor(mic_pressure, dtype=torch.complex64, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg.get("lr", 1e-3)))
    reg_coeff = float(model_cfg.get("reg_coeff", 1e-5))
    iterations = int(model_cfg.get("iterations", 300))
    best_loss = float("inf")
    best_data_loss = float("inf")
    best_state = None
    losses: list[float] = []

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        weights = model.compute_weights()
        prediction = train_matrix @ (weights * ds)
        residual = prediction - target
        data_loss = ((residual.real**2 + residual.imag**2).mean()) / ((target.real**2 + target.imag**2).mean() + 1e-8)
        reg_loss = reg_coeff * (weights.real**2 + weights.imag**2).mean()
        loss = data_loss + reg_loss
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_data_loss = float(data_loss.item())
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    runtime_sec = time.perf_counter() - start

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        source_tensor = torch.tensor(source_points, dtype=torch.float32, device=device)
        estimated = (eval_matrix @ (model.compute_weights_for_coords(source_tensor) * ds)).detach().cpu().numpy().reshape(-1)

    nmse_db, ncc = nmse_db_and_ncc(estimated, ground_truth)
    np.save(estimated_path, estimated)
    np.save(gt_path, ground_truth)
    np.save(output_dir / "eval_coords.npy", np.asarray(case_data["eval_coords"], dtype=np.float64))
    np.save(output_dir / "loss_history.npy", np.asarray(losses, dtype=np.float32))
    torch.save(model.state_dict(), output_dir / "checkpoint.pt")

    metrics = {
        "model_name": model_name,
        "dimension": dimension,
        "nmse_db": nmse_db,
        "ncc": ncc,
        "runtime_sec": float(runtime_sec),
        "param_count": int(param_count),
        "temporal_frequency_hz": freq_hz,
        "bin": int(case_data["bin_index"]),
        "n_mics_fixed": int(case_data["mics"]),
        "iterations": iterations,
        "best_train_loss": best_loss,
        "best_data_loss": best_data_loss,
        "anchors": int(geometry["anchors"]),
        "resolved_device": device,
    }
    write_json(
        output_dir / "config.json",
        {
            "model_name": model_name,
            "arch_cfg": arch_cfg,
            "train_cfg": {
                "lr": float(model_cfg.get("lr", 1e-3)),
                "reg_coeff": reg_coeff,
                "iterations": iterations,
                "device": device_name,
            },
            "geometry": {key: deepcopy(value) for key, value in geometry.items() if key != "point_cloud"},
            "room_cfg": case_data["room_cfg"],
        },
    )
    write_json(metrics_path, metrics)
    write_json(output_dir / "status.json", {"status": "completed", "model_name": model_name})
    return metrics
