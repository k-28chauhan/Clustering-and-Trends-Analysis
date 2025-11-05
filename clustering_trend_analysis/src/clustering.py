import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def estimate_optimal_clusters(
	embeddings: np.ndarray,
	min_clusters: int = 2,
	max_clusters: int = 10,
	random_state: int = 42,
) -> int:
	"""Estimate optimal number of clusters using silhouette score.

	Tests different numbers of clusters and selects the one with the highest
	silhouette score, which measures how well-separated clusters are.

	Parameters
	----------
	embeddings: np.ndarray
		2D array of shape (n_samples, embedding_dim).
	min_clusters: int
		Minimum number of clusters to test.
	max_clusters: int
		Maximum number of clusters to test.
	random_state: int
		Random seed for reproducibility.

	Returns
	-------
	int
		Estimated optimal number of clusters.
	"""
	if embeddings.ndim != 2:
		raise ValueError("embeddings must be a 2D array: (n_samples, embedding_dim)")

	n_samples = embeddings.shape[0]
	max_clusters = min(max_clusters, n_samples - 1)
	min_clusters = max(min_clusters, 2)

	if max_clusters < min_clusters:
		logging.warning(
			"max_clusters (%d) < min_clusters (%d), using min_clusters",
			max_clusters,
			min_clusters,
		)
		return min_clusters

	logging.info(
		"Estimating optimal cluster count using silhouette score (range: %d-%d)",
		min_clusters,
		max_clusters,
	)

	# Normalize embeddings
	scaler = StandardScaler()
	embeddings_scaled = scaler.fit_transform(embeddings)

	cluster_range = range(min_clusters, max_clusters + 1)
	silhouette_scores = []

	for k in cluster_range:
		kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
		labels = kmeans.fit_predict(embeddings_scaled)
		score = silhouette_score(embeddings_scaled, labels)
		silhouette_scores.append(score)

	optimal_k = cluster_range[np.argmax(silhouette_scores)]
	logging.info(
		"Optimal clusters: %d (silhouette score: %.4f)",
		optimal_k,
		max(silhouette_scores),
	)

	return optimal_k


def run_clustering_and_pca(
	embeddings: np.ndarray,
	n_clusters: Optional[int] = None,
	random_state: int = 42,
	auto_cluster_min: int = 2,
	auto_cluster_max: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
	"""Run KMeans clustering and PCA dimensionality reduction to 2D.

	The embeddings are normalized using StandardScaler (mean=0, std=1) before
	clustering and PCA to optimize performance and ensure all features contribute
	equally to the distance calculations.

	If n_clusters is None, the optimal number of clusters will be automatically
	estimated using silhouette score.

	Parameters
	----------
	embeddings: np.ndarray
		2D array of shape (n_samples, embedding_dim).
	n_clusters: Optional[int]
		Number of KMeans clusters. If None, will be automatically estimated using silhouette score.
	random_state: int
		Random seed for reproducibility.
	auto_cluster_min: int
		Minimum number of clusters to test for automatic estimation.
	auto_cluster_max: int
		Maximum number of clusters to test for automatic estimation.

	Returns
	-------
	labels: np.ndarray
		Cluster labels of shape (n_samples,).
	pca_2d: np.ndarray
		PCA-transformed coordinates of shape (n_samples, 2).
	artifacts: Dict[str, object]
		Dictionary with fitted models: {"kmeans": KMeans, "pca": PCA, "scaler": StandardScaler}.
	"""
	if embeddings.ndim != 2:
		raise ValueError("embeddings must be a 2D array: (n_samples, embedding_dim)")

	# Automatically estimate optimal cluster count if not provided
	if n_clusters is None:
		n_clusters = estimate_optimal_clusters(
			embeddings,
			min_clusters=auto_cluster_min,
			max_clusters=auto_cluster_max,
			random_state=random_state,
		)

	logging.info("Normalizing embeddings using StandardScaler")
	scaler = StandardScaler()
	embeddings_scaled = scaler.fit_transform(embeddings)

	logging.info("Running KMeans with n_clusters=%d", n_clusters)
	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
	labels = kmeans.fit_predict(embeddings_scaled)

	logging.info("Running PCA to reduce to 2D for visualization")
	pca = PCA(n_components=2, random_state=random_state)
	pca_2d = pca.fit_transform(embeddings_scaled)

	return labels, pca_2d, {"kmeans": kmeans, "pca": pca, "scaler": scaler}


