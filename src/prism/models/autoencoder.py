"""Deep Autoencoder: neural reconstruction loss (§5.2.2).

Implemented in PyTorch rather than Keras: TensorFlow publishes no distribution
for Python 3.14 (verified against PyPI 2026-08-05), while torch ships a native
cp314 wheel. The architecture and training regime are unchanged from the
proposal; only the framework differs.

Trained on non-fraud rows only. The network learns to compress and rebuild a
"normal" transaction; anything it reconstructs badly is, by construction,
unlike anything it was trained on. Reconstruction MSE is the anomaly score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prism import config
from prism.models.base import ArrayLike, CategoricalExpander, FraudModel, ScoreCalibrator


class _AENet(nn.Module):
    """Symmetric encoder/decoder with a narrow latent bottleneck."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()

        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            encoder_layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderModel(FraudModel):
    name = "autoencoder"

    def __init__(self, **overrides) -> None:
        params = dict(config.get("models.autoencoder"))
        params.update(overrides)
        self.categorical_columns = tuple(params.pop("categorical_columns", ()))
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _AENet | None = None
        self._input_dim: int | None = None
        self._calibrator = ScoreCalibrator()
        self._expander = CategoricalExpander(self.categorical_columns)

    def fit(self, X: ArrayLike, y: pd.Series | np.ndarray | None = None):
        torch.manual_seed(config.seed())
        if isinstance(X, pd.DataFrame):
            self._expander.fit(X)
        data = self._expander.transform(X)

        # Train on normals only: the defining property of this approach.
        train_data = data
        if self.params.get("train_on_normal_only", True):
            if y is None:
                raise ValueError(
                    "train_on_normal_only=True requires y to identify normal rows."
                )
            mask = np.asarray(y).astype(int) == 0
            train_data = data[mask]

        self._input_dim = data.shape[1]
        self._net = _AENet(
            input_dim=self._input_dim,
            hidden_dims=list(self.params["hidden_dims"]),
            latent_dim=int(self.params["latent_dim"]),
        ).to(self.device)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data)),
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        criterion = nn.MSELoss()

        self._net.train()
        for epoch in range(int(self.params["epochs"])):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(batch), batch)
                loss.backward()
                optimizer.step()
                total += loss.item() * batch.size(0)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [autoencoder] epoch {epoch + 1:>3}/{self.params['epochs']}  "
                    f"loss={total / len(loader.dataset):.6f}"
                )

        # Calibrate against the full training distribution (normals AND frauds),
        # so the [0, 1] scale spans the errors actually seen in production.
        self._calibrator.fit(self._reconstruction_error(data))
        return self

    def _reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        assert self._net is not None, "call fit() first"
        self._net.eval()
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(data), 4096):
                chunk = torch.from_numpy(data[start : start + 4096]).to(self.device)
                rebuilt = self._net(chunk)
                errors.append(
                    torch.mean((rebuilt - chunk) ** 2, dim=1).cpu().numpy()
                )
        return np.concatenate(errors) if errors else np.empty(0)

    def risk_score(self, X: ArrayLike) -> np.ndarray:
        return self._calibrator.transform(self._reconstruction_error(self._expander.transform(X)))

    @property
    def expander(self) -> CategoricalExpander:
        return self._expander

    def squared_errors(self, X: ArrayLike) -> np.ndarray:
        """Each expanded column's share of the row's reconstruction error, (rows, width):
        the squared error over the width, so the shares sum to the error the score is
        calibrated from."""
        assert self._net is not None, "call fit() first"
        data = self._expander.transform(X)
        self._net.eval()
        with torch.no_grad():
            rebuilt = self._net(torch.from_numpy(data).to(self.device)).cpu().numpy()
        return ((rebuilt - data) ** 2) / data.shape[1]

    def save(self, model_dir: Path) -> Path:
        assert self._net is not None, "call fit() first"
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self.name}.pt"
        torch.save(self._net.state_dict(), path)
        (model_dir / f"{self.name}.meta.json").write_text(
            json.dumps(
                {
                    "input_dim": self._input_dim,
                    "hidden_dims": list(self.params["hidden_dims"]),
                    "latent_dim": int(self.params["latent_dim"]),
                    "calibrator": self._calibrator.state(),
                    "expander": self._expander.state(),
                }
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, model_dir: Path) -> "AutoencoderModel":
        path = model_dir / f"{cls.name}.pt"
        meta_path = model_dir / f"{cls.name}.meta.json"
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No trained autoencoder at {path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance = cls()
        instance._expander = CategoricalExpander.from_state(meta.get("expander", {}))
        instance.categorical_columns = tuple(instance._expander.columns)
        instance._input_dim = int(meta["input_dim"])
        instance._net = _AENet(
            input_dim=instance._input_dim,
            hidden_dims=list(meta["hidden_dims"]),
            latent_dim=int(meta["latent_dim"]),
        ).to(instance.device)
        # weights_only=True restricts the load to plain tensors, so a tampered
        # checkpoint cannot execute code on load.
        instance._net.load_state_dict(
            torch.load(path, map_location=instance.device, weights_only=True)
        )
        instance._net.eval()
        instance._calibrator = ScoreCalibrator.from_state(meta["calibrator"])
        return instance
