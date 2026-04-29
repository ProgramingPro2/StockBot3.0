"""
Market Regime Specialist: VAE for regime detection.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from ..config import DEVICE, REGIME_MODEL_DIR, VAE_HIDDEN_DIMS, VAE_INPUT_DIM, VAE_LATENT_DIM, VAE_REGIME_CLUSTERS
from ..data.loader import load_ohlcv
from ..data.processor import prepare_regime_features
from ..utils.logger import get_logger

logger = get_logger()

REGIME_NAMES = ["Bull", "Bear", "Sideways", "Crash"]
DEFENSIVE_REGIMES = ["Bear", "Crash"]  # High regime_flag


class VAE(nn.Module):
    """Variational Autoencoder for market regime latent space."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


_vae: VAE | None = None
_kmeans: KMeans | None = None
_regime_labels: list[str] | None = None  # Cluster index -> regime name
_scaler_mean: np.ndarray | None = None
_scaler_std: np.ndarray | None = None


def _get_device() -> str:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _reconstruction_loss(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + 0.1 * kld


def _ensure_model(retrain: bool = False) -> bool:
    """Load or train VAE + KMeans."""
    global _vae, _kmeans, _regime_labels, _scaler_mean, _scaler_std

    model_path = REGIME_MODEL_DIR / "vae.pt"
    kmeans_path = REGIME_MODEL_DIR / "kmeans.pkl"
    meta_path = REGIME_MODEL_DIR / "meta.pkl"

    if model_path.exists() and kmeans_path.exists() and meta_path.exists() and not retrain:
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                input_dim = meta.get("input_dim", VAE_INPUT_DIM)
            _vae = VAE(input_dim, VAE_LATENT_DIM, VAE_HIDDEN_DIMS)
            _vae.load_state_dict(torch.load(model_path, map_location=_get_device()))
            _vae.to(_get_device())
            _vae.eval()
            with open(kmeans_path, "rb") as f:
                _kmeans = pickle.load(f)
            _regime_labels = meta.get("regime_labels", REGIME_NAMES[:VAE_REGIME_CLUSTERS])
            _scaler_mean = meta.get("scaler_mean")
            _scaler_std = meta.get("scaler_std")
            return True
        except Exception as e:
            logger.warning("Failed to load regime model: %s", e)

    # Train on SPY
    from ..config import YFINANCE_LOOKBACK_YEARS
    spy = load_ohlcv("SPY", lookback_years=YFINANCE_LOOKBACK_YEARS)
    if spy.empty or len(spy) < 500:
        logger.warning("Insufficient SPY data for regime VAE")
        return False

    feats = prepare_regime_features(spy)
    if feats.empty or len(feats) < 200:
        return False

    n_features = feats.shape[1]
    input_dim = n_features
    feats = feats.values

    # Normalize
    _scaler_mean = np.nanmean(feats, axis=0)
    _scaler_std = np.nanstd(feats, axis=0)
    _scaler_std[_scaler_std == 0] = 1
    X = (feats - _scaler_mean) / _scaler_std
    X = np.nan_to_num(X, nan=0.0)

    dataset = torch.FloatTensor(X)
    _bs = int(os.getenv("REGIME_VAE_BATCH_SIZE", "64").strip() or "64")
    loader = torch.utils.data.DataLoader(dataset, batch_size=_bs, shuffle=True)

    vae = VAE(input_dim, VAE_LATENT_DIM, VAE_HIDDEN_DIMS)
    vae.to(_get_device())
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    from tqdm import tqdm
    _epochs = int(os.getenv("REGIME_VAE_EPOCHS", "50").strip() or "50")
    for epoch in tqdm(range(_epochs), desc="Regime VAE", unit="epoch"):
        for batch in loader:
            batch = batch.to(_get_device())
            opt.zero_grad()
            recon, mu, logvar = vae(batch)
            loss = _reconstruction_loss(batch, recon, mu, logvar)
            loss.backward()
            opt.step()

    vae.eval()
    with torch.no_grad():
        _, mu, _ = vae(torch.FloatTensor(X).to(_get_device()))
        z = mu.cpu().numpy()

    kmeans = KMeans(n_clusters=VAE_REGIME_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z)

    # Map clusters to regime names by centroid characteristics
    centroids = kmeans.cluster_centers_
    regime_map = []
    for i in range(VAE_REGIME_CLUSTERS):
        if centroids[i, 0] > 0.1 and np.std(centroids[i]) < 0.5:
            regime_map.append("Bull")
        elif centroids[i, 0] < -0.1 and np.std(centroids[i]) > 0.5:
            regime_map.append("Crash")
        elif centroids[i, 0] < -0.05:
            regime_map.append("Bear")
        else:
            regime_map.append("Sideways")
    _regime_labels = regime_map

    REGIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), model_path)
    with open(kmeans_path, "wb") as f:
        pickle.dump(kmeans, f)
    with open(meta_path, "wb") as f:
        pickle.dump({
            "regime_labels": _regime_labels,
            "scaler_mean": _scaler_mean,
            "scaler_std": _scaler_std,
            "input_dim": input_dim,
        }, f)

    _vae = vae
    _kmeans = kmeans
    return True


def detect_regime(spy_data: pd.DataFrame | None = None) -> dict:
    """
    Detect current market regime from SPY data.
    Returns {"regime": str, "flag": float} where flag in [0, 1] for defensive posture.
    1 = Crash/Bear (defensive), 0 = Bull (aggressive).
    """
    if not _ensure_model():
        return {"regime": "Sideways", "flag": 0.5}

    if spy_data is None:
        spy_data = load_ohlcv("SPY")
    if spy_data.empty:
        return {"regime": "Sideways", "flag": 0.5}

    feats = prepare_regime_features(spy_data)
    if feats.empty:
        return {"regime": "Sideways", "flag": 0.5}

    n_features = feats.shape[1]
    input_dim = len(_scaler_mean) if _scaler_mean is not None else min(VAE_INPUT_DIM, n_features)
    X = feats.values[:, :min(input_dim, n_features)]
    if X.shape[1] < input_dim:
        X = np.hstack([X, np.zeros((len(X), input_dim - X.shape[1]))])

    X = (X - _scaler_mean) / _scaler_std
    X = np.nan_to_num(X, nan=0.0)
    X = torch.FloatTensor(X[-1:]).to(_get_device())

    _vae.eval()
    with torch.no_grad():
        recon, mu, _ = _vae(X)
        recon_err = float(torch.mean((recon - X) ** 2).cpu())
        z = mu.cpu().numpy()

    cluster = _kmeans.predict(z)[0]
    regime = _regime_labels[cluster] if cluster < len(_regime_labels) else "Sideways"

    # Flag: 1 for Bear/Crash, 0 for Bull, 0.5 for Sideways
    if regime in DEFENSIVE_REGIMES:
        flag = 0.9 if regime == "Crash" else 0.7
    elif regime == "Bull":
        flag = 0.0
    else:
        flag = 0.5

    # Boost flag if reconstruction error is high (anomaly)
    if recon_err > 0.5:
        flag = min(1.0, flag + 0.2)

    return {"regime": regime, "flag": flag}
