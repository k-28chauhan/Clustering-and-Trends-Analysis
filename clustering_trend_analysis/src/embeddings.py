"""
Enhanced generate_embeddings() for academic research papers.

Backwards-compatible with previous version — caching, normalization,
and reduction logic remain unchanged.

Added:
 - Field weighting (title, abstract, findings, limitations)
 - Per-sentence pooling (better for long abstracts)
 - Domain model default: allenai/specter
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Optional imports for extra features
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
    model_name: str = "allenai/specter",  # ✅ Default scientific-domain model
    batch_size: int = 32,
    device: Optional[str] = None,
    normalize: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
    overwrite_cache: bool = False,
    reduce_to: Optional[int] = None,
    use_fp16: bool = False,
    verbose: bool = False,
    field_weights: Optional[dict] = None,  # e.g., {"title":3.0,"abstract":2.0,"findings":1.5,"limitations":1.0}
    per_sentence_pooling: bool = True,
) -> np.ndarray:
    """
    Generate optimized embeddings for academic papers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the text columns.
    text_columns : Sequence[str]
        Columns to use (e.g., ["title","abstract","findings","limitations"])
    model_name : str
        SentenceTransformer model identifier. Default: 'allenai/specter'
    batch_size : int
        Batch size for encoding.
    device : Optional[str]
        'cpu' or 'cuda'. Auto-detects if None.
    normalize : bool
        Normalize embeddings to unit length.
    cache_path : Optional[str | Path]
        If provided, will save/load cached embeddings (.npy).
    overwrite_cache : bool
        Recompute even if cache exists.
    reduce_to : Optional[int]
        Optionally reduce embedding dimensionality via TruncatedSVD.
    use_fp16 : bool
        Use float16 precision for faster GPU inference.
    verbose : bool
        Enable extra logging.
    field_weights : Optional[dict]
        Weight per text column. Default: automatic heuristic.
    per_sentence_pooling : bool
        Split long texts into sentences and average their embeddings.
    """
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)

    # Validate columns
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns in DataFrame: {missing}")

    # Default field weights
    if field_weights is None:
        field_weights = {col: 1.0 for col in text_columns}
        for key in field_weights:
            if "title" in key.lower():
                field_weights[key] = 3.0
            elif "abstract" in key.lower():
                field_weights[key] = 2.0
            elif "finding" in key.lower():
                field_weights[key] = 1.5
            elif "limitation" in key.lower():
                field_weights[key] = 1.0

    # Determine device
    if device is None:
        device = _auto_device() or "cpu"
    if verbose:
        logger.info("Using device=%s (torch available=%s)", device, torch is not None)

    # Cache loading
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite_cache:
            logger.info("Loading cached embeddings from %s", cache_path)
            arr = np.load(cache_path)
            if reduce_to is not None and arr.shape[1] != reduce_to:
                logger.info("Cached embeddings dimension mismatch; recomputing")
            elif normalize and not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-3):
                logger.info("Cached embeddings not normalized; recomputing")
            else:
                return arr

    # Load SentenceTransformer model
    load_kwargs = {}
    if use_fp16 and device == "cuda" and torch is not None:
        try:
            load_kwargs["torch_dtype"] = torch.float16
        except Exception:
            load_kwargs = {}

    logger.info("Loading SentenceTransformer model '%s' on device=%s", model_name, device)
    try:
        model = SentenceTransformer(model_name, device=device, **load_kwargs)
    except TypeError:
        model = SentenceTransformer(model_name, device=device)

    # Sentence splitting helper
    import re
    sentence_splitter = lambda txt: [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if s.strip()]

    all_embeddings = []
    for idx, row in df[text_columns].fillna("").astype(str).iterrows():
        field_embs = []
        total_weight = 0.0

        for col in text_columns:
            text = row[col].strip()
            weight = float(field_weights.get(col, 1.0))
            if not text:
                continue

            if per_sentence_pooling:
                sentences = sentence_splitter(text)
                if not sentences:
                    sentences = [text]
                sent_embs = model.encode(
                    sentences,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )
                field_emb = np.mean(sent_embs, axis=0)
            else:
                field_emb = model.encode(
                    [text],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )[0]

            field_embs.append(field_emb * weight)
            total_weight += weight

        if field_embs:
            doc_emb = np.sum(np.vstack(field_embs), axis=0) / total_weight
        else:
            doc_emb = np.zeros(model.get_sentence_embedding_dimension())
        all_embeddings.append(doc_emb)

    embeddings = np.vstack(all_embeddings)

    # Normalization
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    # Dimensionality reduction (unchanged)
    if reduce_to is not None:
        if TruncatedSVD is None:
            logger.warning("Requested reduce_to=%s but sklearn.decomposition.TruncatedSVD unavailable", reduce_to)
        else:
            cur_dim = embeddings.shape[1]
            if reduce_to < cur_dim:
                svd = TruncatedSVD(n_components=reduce_to, random_state=42)
                embeddings = svd.fit_transform(embeddings)
                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embeddings = embeddings / norms

    # Save cache
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info("Saved embeddings to cache %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save embeddings to cache %s: %s", cache_path, e)

    return embeddings
