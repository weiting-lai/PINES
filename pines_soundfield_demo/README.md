# PINES Sound Field Demo

Standalone 2D and 3D sound field demos for two PINES variants:

- `pines_tanh`
- `pines_siren`

Each run trains the selected PINES models and writes sound field figures, metrics, raw pressure arrays, and a manifest.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python run_case_2d.py --config config_2d.yaml --pines-config config_pines.yaml --device cpu
python run_case_3d.py --config config_3d.yaml --pines-config config_pines.yaml --device cpu
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Outputs

Default outputs are written under:

```text
output/2d/<case_name>/
output/3d/<case_name>/
```

Each case contains:

- `figures/fig_pines_tanh.png`
- `figures/fig_pines_siren.png`
- `soundfield_manifest.json`
- `raw/<model>/metrics.json`
- `raw/<model>/config.json`
- `raw/<model>/pressure_estimated.npy`
- `raw/<model>/pressure_gt.npy`

## Configs

- `config_2d.yaml` controls the 2D room, region of interest, grid, microphones, frequency bin, source count, T60, seed, and output root.
- `config_3d.yaml` controls the 3D room, XY slice, 3D microphone candidate grid, frequency bin, source count, T60, seed, and output root.
- `config_pines.yaml` controls PINES hyperparameters, device, microphone SNR, and equivalent source geometry.

The defaults are intentionally small so the demo can run on CPU. For cleaner figures, increase:

- `models.*.iterations` to `1000` or higher
- `equivalent_sources_2d.anchors` to `160` or higher
- `equivalent_sources_3d.anchors` to `192` or higher
- `n_grid` in the case config

## CLI

```bash
python run_case_2d.py --output output/2d_custom --device auto --models pines_tanh pines_siren
python run_case_3d.py --output output/3d_custom --device cuda --models pines_siren
```

Both runners support:

- `--output`
- `--device auto|cpu|cuda`
- `--models pines_tanh pines_siren`
- `--resume`

## Notes

The default configs use `t60: 0.0`, which runs an anechoic direct-path simulation. If you set `t60 > 0`, the demo uses `pyroomacoustics` for image-source room simulation.

## Attribution

Small utility routines for metrics/noise follow the style of MIT-licensed helper code from:

- `PIDL-sound-field-reconstruction`: https://github.com/steDamiano/PIDL-sound-field-reconstruction
- `local_soundfield_reconstruction`: https://github.com/manvhah/local_soundfield_reconstruction
