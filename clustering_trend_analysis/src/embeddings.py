import logging
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_embeddings(
	df: pd.DataFrame,
	text_columns: Sequence[str],
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
	batch_size: int = 32,
	device: Optional[str] = None,
) -> np.ndarray:
	"""Generate dense embeddings for concatenated text columns using SentenceTransformer.

	Parameters
	----------
	df: pd.DataFrame
		Input DataFrame containing the text columns.
	text_columns: Sequence[str]
		Columns to concatenate into a single text per row.
	model_name: str
		SentenceTransformer model identifier.
	batch_size: int
		Batch size for encoding.
	device: Optional[str]
		Optional device override (e.g., "cpu", "cuda").

	Returns
	-------
	np.ndarray
		2D array of shape (n_samples, embedding_dim).
	"""
	missing = [c for c in text_columns if c not in df.columns]
	if missing:
		raise ValueError(f"Missing text columns in DataFrame: {missing}")

	# Combine text fields; handle missing values gracefully
	texts: List[str] = (
		df[list(text_columns)]
		.fillna("")
		.astype(str)
		.apply(lambda r: " | ".join(s.strip() for s in r if s), axis=1)
		.tolist()
	)

	logging.info(
		"Loading SentenceTransformer model '%s' on device=%s",
		model_name,
		device if device is not None else "auto",
	)
	model = SentenceTransformer(model_name, device=device)

	logging.info("Encoding %d texts into embeddings", len(texts))
	embeddings = model.encode(
		texts,
		batch_size=batch_size,
		show_progress_bar=True,
		normalize_embeddings=True,
	)

	return np.asarray(embeddings)


