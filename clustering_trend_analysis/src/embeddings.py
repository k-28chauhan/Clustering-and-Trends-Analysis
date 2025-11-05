"""
Enhanced generate_embeddings() for clustering_trend_analysis.

Handles stale caches safely:
 - Detects when cached embeddings don’t match current dataset
 - Automatically regenerates and overwrites cache
 - Fully backwards compatible
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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


def _dataset_hash(df: pd.DataFrame, text_columns: Sequence[str]) -> str:
    """Compute a short hash fingerprint of the dataset based on selected columns."""
    # Using a sample of rows for efficiency
    sample_bytes = df[text_columns].head(1000).to_csv(index=False).encode("utf-8")
    return hashlib.md5(sample_bytes).hexdigest()[:8]


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
    Automatically handles stale or mismatched cache files.
    """ 
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)

    # ✅ Validate text columns
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns in DataFrame: {missing}")

    # ✅ Build text list
    texts: List[str] = (
        df[list(text_columns)]
        .fillna("")
        .astype(str)
        .apply(lambda r: " | ".join(s.strip() for s in r if s), axis=1)
        .tolist()
    )

    # ✅ Determine device
    if device is None:
        device = _auto_device() or "cpu"
    if verbose:
        logger.info("Using device=%s (torch available=%s)", device, torch is not None)

    # ✅ Generate automatic cache filename if not provided
    if cache_path is None:
        dataset_id = _dataset_hash(df, text_columns)
        cache_name = f"embeddings_{Path(model_name).stem}_{dataset_id}.npy"
        cache_path = Path("outputs") / cache_name

    cache_path = Path(cache_path)
    should_recompute = overwrite_cache

    # ✅ Attempt to load cache safely
    if cache_path.exists() and not overwrite_cache:
        try:
            logger.info("Loading cached embeddings from %s", cache_path)
            arr = np.load(cache_path)
            # --- Validate cache integrity ---
            if arr.shape[0] != len(df):
                logger.warning(
                    "Cached embeddings rows (%d) != dataset rows (%d); marking cache as stale",
                    arr.shape[0],
                    len(df),
                )
                should_recompute = True
            elif reduce_to is not None and arr.shape[1] != reduce_to:
                logger.warning(
                    "Cached embedding dim (%d) != requested reduce_to (%d); marking cache as stale",
                    arr.shape[1],
                    reduce_to,
                )
                should_recompute = True
            elif normalize and not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-3):
                logger.warning("Cached embeddings not normalized; marking cache as stale")
                should_recompute = True
            else:
                # Cache valid
                logger.info("Cache validated — using cached embeddings.")
                return arr
        except Exception as e:
            logger.warning("Failed to load cache (%s); will recompute: %s", cache_path, e)
            should_recompute = True

    # ✅ Recompute embeddings if cache invalid
    if should_recompute or not cache_path.exists():
        logger.info("Generating new embeddings using model '%s'...", model_name)
        load_kwargs = {}
        if use_fp16 and device == "cuda" and torch is not None:
            try:
                load_kwargs["torch_dtype"] = torch.float16
                if verbose:
                    logger.info("Attempting to load model in float16 mode")
            except Exception as e:
                logger.warning("Failed to set fp16: %s; fallback to default", e)
                load_kwargs = {}

        model = SentenceTransformer(model_name, device=device, **load_kwargs)
        logger.info("Encoding %d texts (batch_size=%d)", len(texts), batch_size)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
        embeddings = np.asarray(embeddings)

        # ✅ Optional dimensionality reduction
        if reduce_to is not None:
            if TruncatedSVD is None:
                logger.warning("TruncatedSVD unavailable; skipping reduction")
            else:
                cur_dim = embeddings.shape[1]
                if reduce_to < cur_dim:
                    logger.info("Reducing dimensionality %d → %d via TruncatedSVD", cur_dim, reduce_to)
                    svd = TruncatedSVD(n_components=reduce_to, random_state=42)
                    embeddings = svd.fit_transform(embeddings)
                    if normalize:
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        embeddings = embeddings / norms
                else:
                    logger.info("reduce_to >= current dim; skipping reduction")

        # ✅ Save to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info("Saved embeddings to cache %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save embeddings cache: %s", e)

        return embeddings

    # Should not reach here, but return dummy if logic fails
    logger.warning("Unexpected cache logic path — recomputing as fallback.")
    return generate_embeddings(
        df,
        text_columns=text_columns,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        cache_path=cache_path,
        overwrite_cache=True,
        reduce_to=reduce_to,
        use_fp16=use_fp16,
        verbose=verbose,
    )
