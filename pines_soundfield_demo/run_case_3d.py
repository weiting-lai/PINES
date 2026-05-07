from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.demo_core import ALLOWED_MODELS, make_3d_case, run_root_name, train_pines_model
from src.plotting import plot_soundfield
from src.utils import load_yaml, write_json


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _selected_models(models: list[str] | None) -> tuple[str, ...]:
    return tuple(models) if models else ALLOWED_MODELS


def main(
    config_path: Path,
    pines_config_path: Path,
    output: Path | None,
    device: str | None,
    models: tuple[str, ...],
    resume: bool,
) -> Path:
    case_cfg = load_yaml(_resolve(config_path))
    pines_cfg = load_yaml(_resolve(pines_config_path))
    if output is not None:
        case_cfg["output_root"] = str(output)
    if device is not None:
        pines_cfg["device"] = device

    case_data = make_3d_case(case_cfg)
    anchors = int(pines_cfg.get("equivalent_sources_3d", {}).get("anchors", 96))
    output_root = _resolve(Path(case_cfg.get("output_root", "output/3d")))
    run_dir = output_root / run_root_name(
        seed=int(case_cfg.get("seed", 42)),
        fold_idx=int(case_cfg.get("fold", 0)),
        bin_index=int(case_cfg.get("frequency_bin", 40)),
        mics=int(case_cfg.get("mics", 24)),
        anchors=anchors,
        case_tag=case_cfg.get("case_tag"),
        extra=f"z{float(case_cfg.get('slice_z', 0.0)):.2f}".replace("-", "m").replace(".", "p"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict[str, Any]] = {}
    figure_paths: dict[str, str] = {}
    first_gt = np.asarray(case_data["ground_truth"]).reshape(-1)
    amplitude = float(np.max(np.abs(np.real(first_gt))))
    field_clim = (-amplitude, amplitude)

    for model_name in models:
        if model_name not in ALLOWED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        model_dir = run_dir / "raw" / model_name
        metrics = train_pines_model(
            model_name=model_name,
            case_data=case_data,
            pines_cfg=pines_cfg,
            output_dir=model_dir,
            device_name=str(pines_cfg.get("device", "auto")),
            resume=resume,
        )
        fig_path = run_dir / "figures" / f"fig_{model_name}.png"
        plot_soundfield(
            model_name=model_name,
            true_field=np.load(model_dir / "pressure_gt.npy"),
            approx_field=np.load(model_dir / "pressure_estimated.npy"),
            n_grid=int(case_data["n_grid"]),
            region_size=case_data["room_cfg"]["region_size"],
            output_path=fig_path,
            mic_positions=np.asarray(case_data["mic_coords"], dtype=np.float64),
            field_clim=field_clim,
        )
        figure_paths[model_name] = fig_path.as_posix()
        payloads[model_name] = {
            "metrics": metrics,
            "raw_dir": model_dir.as_posix(),
            "metrics_path": (model_dir / "metrics.json").as_posix(),
            "config_path": (model_dir / "config.json").as_posix(),
            "pressure_estimated_path": (model_dir / "pressure_estimated.npy").as_posix(),
            "pressure_gt_path": (model_dir / "pressure_gt.npy").as_posix(),
            "eval_coords_path": (model_dir / "eval_coords.npy").as_posix(),
            "figure_path": fig_path.as_posix(),
        }

    manifest = {
        "scenario": {
            "kind": "3d_slice_soundfield_demo",
            "seed": int(case_cfg.get("seed", 42)),
            "fold": int(case_cfg.get("fold", 0)),
            "fold_seed": int(case_data["fold_seed"]),
            "bin": int(case_data["bin_index"]),
            "frequency_hz": float(case_data["frequency_hz"]),
            "mics": int(case_data["mics"]),
            "n_grid": int(case_data["n_grid"]),
            "slice_z": float(case_data["slice_z"]),
        },
        "room_cfg": case_data["room_cfg"],
        "source_positions": np.asarray(case_data["source_positions"], dtype=np.float64),
        "mic_positions_centered": np.asarray(case_data["mic_coords"], dtype=np.float64),
        "models": payloads,
        "figures": figure_paths,
        "run_root": run_dir.as_posix(),
    }
    write_json(run_dir / "soundfield_manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the standalone 3D PINES sound field demo.")
    parser.add_argument("--config", type=Path, default=Path("config_3d.yaml"))
    parser.add_argument("--pines-config", type=Path, default=Path("config_pines.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--models", nargs="+", choices=list(ALLOWED_MODELS), default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = main(
        config_path=args.config,
        pines_config_path=args.pines_config,
        output=args.output,
        device=args.device,
        models=_selected_models(args.models),
        resume=bool(args.resume),
    )
    print(f"Done. Outputs: {out}")
