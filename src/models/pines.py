from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_levels: int = 0, coord_scale: float = 2.0, coord_dim: int = 2):
        super().__init__()
        self.num_levels = int(num_levels)
        self.coord_scale = float(coord_scale)
        self.coord_dim = int(coord_dim)
        if self.num_levels > 0:
            freqs = torch.pow(2.0, torch.arange(self.num_levels, dtype=torch.float32)) * math.pi
            self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        if self.num_levels <= 0:
            return self.coord_dim
        return self.coord_dim * (1 + 2 * self.num_levels)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_norm = coords / self.coord_scale
        if self.num_levels <= 0:
            return coords_norm

        scaled = coords_norm.unsqueeze(-1) * self.freqs
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = encoded.reshape(*coords_norm.shape[:-1], self.coord_dim * 2 * self.num_levels)
        return torch.cat([coords_norm, encoded], dim=-1)


class CoordinateWeightModel(nn.Module):
    def __init__(self, point_cloud: np.ndarray):
        super().__init__()
        points = torch.tensor(point_cloud, dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] not in {2, 3}:
            raise ValueError(f"point_cloud must have shape (N, 2) or (N, 3), got {tuple(points.shape)}")
        self.register_buffer("point_cloud", points)
        self.coord_dim = int(points.shape[1])

    def compute_weights(self) -> torch.Tensor:
        return self.compute_weights_for_coords(self.point_cloud)

    def compute_weights_for_coords(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.forward(coords)
        return torch.complex(output[:, 0], output[:, 1])


class BaselineTanhSharedModel(CoordinateWeightModel):
    def __init__(
        self,
        point_cloud: np.ndarray,
        hidden_dim: int,
        hidden_layers: int,
        pe_levels: int,
        coord_scale: float,
    ):
        super().__init__(point_cloud)
        self.encoder = PositionalEncoding(pe_levels, coord_scale, self.coord_dim)
        layers: list[nn.Module] = [nn.Linear(self.encoder.output_dim, int(hidden_dim)), nn.Tanh()]
        for _ in range(max(0, int(hidden_layers) - 1)):
            layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh()])
        layers.append(nn.Linear(int(hidden_dim), 2))
        self.net = nn.Sequential(*layers)
        self._init_linear()

    def _init_linear(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoder(coords))


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w0: float, is_first: bool):
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features))
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SirenSharedModel(CoordinateWeightModel):
    def __init__(
        self,
        point_cloud: np.ndarray,
        hidden_dim: int,
        hidden_layers: int,
        coord_scale: float,
        w0: float,
    ):
        super().__init__(point_cloud)
        self.coord_scale = float(coord_scale)
        hidden_dim = int(hidden_dim)
        hidden_layers = int(hidden_layers)
        self.input_layer = SineLayer(self.coord_dim, hidden_dim, w0=w0, is_first=True)
        self.hidden_layers = nn.ModuleList(
            [SineLayer(hidden_dim, hidden_dim, w0=w0, is_first=False) for _ in range(max(0, hidden_layers - 1))]
        )
        self.output_layer = nn.Linear(hidden_dim, 2)
        bound = math.sqrt(6.0 / hidden_dim) / max(float(w0), 1.0)
        with torch.no_grad():
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords / self.coord_scale
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


def build_pines_model(arch_cfg: dict[str, Any], point_cloud: np.ndarray) -> CoordinateWeightModel:
    family = str(arch_cfg["family"])
    hidden_dim = int(arch_cfg.get("hidden_dim", 64))
    hidden_layers = int(arch_cfg.get("hidden_layers", 2))
    coord_scale = float(arch_cfg.get("coord_scale", 2.0))

    if family == "baseline_tanh_shared":
        return BaselineTanhSharedModel(
            point_cloud=point_cloud,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            pe_levels=int(arch_cfg.get("pe_levels", 0)),
            coord_scale=coord_scale,
        )
    if family == "siren_shared":
        return SirenSharedModel(
            point_cloud=point_cloud,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            coord_scale=coord_scale,
            w0=float(arch_cfg.get("w0", 30.0)),
        )
    raise ValueError(f"Unknown PINES family: {family}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
