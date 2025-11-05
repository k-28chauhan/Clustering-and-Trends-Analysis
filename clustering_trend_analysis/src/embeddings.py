# embeddings.py
"""
Enhanced generate_embeddings() for clustering_trend_analysis.

Backwards-compatible: default behavior matches previous implementation
(concatenate text columns, encode with SentenceTransformer, normalize, return np.ndarray).

Added optional, non-breaking features:
 - device autodetection (cuda if available)
 - cache_path: save/load embeddings to disk (numpy .npy)
 - overwrite_cache: force recompute
 - reduce_to: optional dimensionality reduction (int) using TruncatedSVD (keeps default behavior if None)
 - use_fp16: best-effort float16 model load on GPU (disabled by default)
 - verbose: toggles extra logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Optional imports for extra features (soft dependencies)
try:
    from sklearn.decomposition import TruncatedSVD
except Exception:
    TruncatedSVD = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore


def _auto_device() -> Optional[str]:
    """Return 'cuda' if available, else 'cpu' (or None if torch not installed)."""
    if torch is None:
        return None
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_embeddings(
    df: pd.DataFrame,
    text_columns: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
    normalize: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
    overwrite_cache: bool = False,
    reduce_to: Optional[int] = None,
    use_fp16: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate dense embeddings for concatenated text columns using SentenceTransformer.

    Backwards-compatible defaults: identical to the previous implementation (concatenate,
    encode, normalize).

    New optional arguments:
      - device: "cpu" or "cuda". If None, autodetects if torch is available.
      - cache_path: path to save/load .npy cache of embeddings.
      - overwrite_cache: if True, recompute even if cache exists.
      - reduce_to: if set to int (e.g., 128), reduce embeddings to that dimension using TruncatedSVD.
      - use_fp16: if True and GPU available, attempt to load model in float16 (best-effort).
      - verbose: extra logging.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the text columns.
    text_columns : Sequence[str]
        Columns to concatenate into a single text per row.
    model_name : str
        SentenceTransformer model identifier.
    batch_size : int
        Batch size for encoding.
    device : Optional[str]
        Optional device override (e.g., "cpu", "cuda"). If None, auto-detects.
    normalize : bool
        If True, normalize embeddings to unit length (recommended for cosine).
    cache_path : Optional[str | Path]
        If provided, will load embeddings from this .npy file if present (unless overwrite_cache=True),
        and will save computed embeddings there.
    overwrite_cache : bool
        Force recomputation even if cache exists.
    reduce_to : Optional[int]
        If provided, reduce embedding dimensionality to this integer using TruncatedSVD.
        Requires sklearn. If not available, reduction is skipped and a warning is logged.
    use_fp16 : bool
        If True and GPU is available, attempts to load model with torch.float16 dtype.
        If unsupported, falls back to default loading.
    verbose : bool
        Extra logging.

    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, embedding_dim) or (n_samples, reduce_to) if reduction used.
    """
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)

    # Validate text columns
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns in DataFrame: {missing}")

    # Build text list
    texts: List[str] = (
        df[list(text_columns)]
        .fillna("")
        .astype(str)
        .apply(lambda r: " | ".join(s.strip() for s in r if s), axis=1)
        .tolist()
    )

    # Determine device
    if device is None:
        device = _auto_device() or "cpu"
    if verbose:
        logger.info("Using device=%s (torch available=%s)", device, torch is not None)

    # Cache handling
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite_cache:
            logger.info("Loading cached embeddings from %s", cache_path)
            arr = np.load(cache_path)
            # If normalization or reduction is requested but cache may not satisfy it,
            # we do a quick check: if reduce_to is set but arr has different dim, recompute.
            if reduce_to is not None and arr.shape[1] != reduce_to:
                logger.info("Cached embeddings dimension (%d) != requested reduce_to (%d); recomputing", arr.shape[1], reduce_to)
            elif normalize and not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-3):
                # cached embeddings not normalized; recompute to ensure consistency
                logger.info("Cached embeddings not normalized (or tolerance exceeded); recomputing")
            else:
                # Cache seems valid; return it directly
                return arr

    # Model loading (best-effort fp16 support)
    load_kwargs = {}
    if use_fp16 and device == "cuda" and torch is not None:
        try:
            load_kwargs["torch_dtype"] = torch.float16
            # Some environments may require device_map; SentenceTransformer will handle device if given
            if verbose:
                logger.info("Attempting to load model with float16 dtype for faster GPU inference")
        except Exception as e:
            logger.warning("Failed to set fp16 dtype: %s. Falling back to default dtype.", e)
            load_kwargs = {}

    logger.info("Loading SentenceTransformer model '%s' on device=%s", model_name, device)
    try:
        model = SentenceTransformer(model_name, device=device, **load_kwargs)
    except TypeError:
        # Older sentence-transformers may not accept torch_dtype in constructor; fall back gracefully
        model = SentenceTransformer(model_name, device=device)

    # Encode
    logger.info("Encoding %d texts into embeddings (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings)

    # Optional dimensionality reduction
    if reduce_to is not None:
        if TruncatedSVD is None:
            logger.warning(
                "Requested reduce_to=%s but sklearn.decomposition.TruncatedSVD is not available; skipping reduction",
                reduce_to,
            )
        else:
            cur_dim = embeddings.shape[1]
            if reduce_to >= cur_dim:
                logger.info("reduce_to (%d) >= current dim (%d); skipping reduction", reduce_to, cur_dim)
            else:
                logger.info("Reducing embeddings dimensionality %d -> %d using TruncatedSVD", cur_dim, reduce_to)
                svd = TruncatedSVD(n_components=reduce_to, random_state=42)
                embeddings = svd.fit_transform(embeddings)
                # If normalization requested originally, re-normalize after reduction
                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embeddings = embeddings / norms

    # Save cache if requested
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info("Saved embeddings to cache %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save embeddings to cache %s: %s", cache_path, e)

    return embeddings
