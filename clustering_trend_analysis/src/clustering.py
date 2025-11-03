import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def run_clustering_and_pca(
	embeddings: np.ndarray,
	n_clusters: int,
	random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
	"""Run KMeans clustering and PCA dimensionality reduction to 2D.

	Parameters
	----------
	embeddings: np.ndarray
		2D array of shape (n_samples, embedding_dim).
	n_clusters: int
		Number of KMeans clusters.
	random_state: int
		Random seed for reproducibility.

	Returns
	-------
	labels: np.ndarray
		Cluster labels of shape (n_samples,).
	pca_2d: np.ndarray
		PCA-transformed coordinates of shape (n_samples, 2).
	artifacts: Dict[str, object]
		Dictionary with fitted models: {"kmeans": KMeans, "pca": PCA}.
	"""
	if embeddings.ndim != 2:
		raise ValueError("embeddings must be a 2D array: (n_samples, embedding_dim)")

	logging.info("Running KMeans with n_clusters=%d", n_clusters)
	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
	labels = kmeans.fit_predict(embeddings)

	logging.info("Running PCA to reduce to 2D for visualization")
	pca = PCA(n_components=2, random_state=random_state)
	pca_2d = pca.fit_transform(embeddings)

	return labels, pca_2d, {"kmeans": kmeans, "pca": pca}


