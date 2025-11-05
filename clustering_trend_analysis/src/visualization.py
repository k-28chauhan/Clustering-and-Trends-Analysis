import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px


def create_interactive_visualizations(
	df: pd.DataFrame,
	pca_2d: np.ndarray,
	output_dir: Path,
) -> Dict[str, Path]:
	"""Create and save interactive cluster scatter and yearly trend plots.

	Parameters
	----------
	df: pd.DataFrame
		DataFrame containing at least columns [title, year, cluster].
	pca_2d: np.ndarray
		2D coordinates from PCA of shape (n_samples, 2).
	output_dir: Path
		Directory to save the HTML plots.

	Returns
	-------
	Dict[str, Path]
		Mapping of plot names to saved file paths.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	if "cluster" not in df.columns:
		raise ValueError("DataFrame must contain 'cluster' column for visualization")

	plot_paths: Dict[str, Path] = {}

	# Scatter plot in PCA space colored by cluster
	scatter_df = df.copy()
	scatter_df["pca_x"] = pca_2d[:, 0]
	scatter_df["pca_y"] = pca_2d[:, 1]

	# Build hover_data dynamically from available columns
	hover_columns = {}
	for col in df.columns:
		if col not in ["pca_x", "pca_y"]:  # Exclude PCA columns from hover
			hover_columns[col] = True

	fig_scatter = px.scatter(
		scatter_df,
		x="pca_x",
		y="pca_y",
		color=scatter_df["cluster"].astype(str),
		hover_data=hover_columns,
		title="Clusters in 2D PCA Space",
	)

	fig_scatter.update_layout(legend_title_text="Cluster")
	scatter_path = output_dir / "clusters_scatter.html"
	fig_scatter.write_html(str(scatter_path), include_plotlyjs="cdn")
	plot_paths["scatter"] = scatter_path
	logging.info("Saved scatter plot to %s", scatter_path)

	# Yearly trend: number of publications per cluster per year (if year column exists)
	# Yearly trend: number of publications per cluster per year (if year column exists)
	year_col = None
	for c in ["year", "Year"]:
		if c in df.columns:
			year_col = c
			break

	if year_col:
		trend_df = (
			df.assign(year=pd.to_numeric(df[year_col], errors="coerce"))
			.dropna(subset=["year"])
			.groupby(["year", "cluster"], dropna=False)
			.size()
			.reset_index(name="count")
		)

		if len(trend_df) > 0:
			fig_trend = px.line(
				trend_df.sort_values(["year", "cluster"]),
				x="year",
				y="count",
				color=trend_df["cluster"].astype(str),
				hover_data={"cluster": True, "year": True, "count": True},
				title="Yearly Trend by Cluster",
				markers=True,
			)
			fig_trend.update_layout(legend_title_text="Cluster")
			trend_path = output_dir / "yearly_trend.html"
			fig_trend.write_html(str(trend_path), include_plotlyjs="cdn")
			plot_paths["trend"] = trend_path
			logging.info("Saved yearly trend plot to %s", trend_path)
		else:
			logging.warning("No valid year data found, skipping yearly trend plot")
	else:
		logging.info("No 'year' column found, skipping yearly trend plot")


	return plot_paths


