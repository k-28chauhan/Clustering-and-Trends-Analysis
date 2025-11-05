import logging
from pathlib import Path
<<<<<<< HEAD
from typing import Dict
=======
from typing import Dict, Optional
>>>>>>> 65ec2bd (Updated Visualization.py)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_interactive_visualizations(
    df: pd.DataFrame,
    pca_2d: np.ndarray,
    output_dir: Path,
    save_html: bool = True,
    template: str = "plotly_white",
) -> Dict[str, Path]:
    """
    Create and save interactive visualizations for clusters and yearly trends.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least [title, year, cluster].
    pca_2d : np.ndarray
        2D coordinates from PCA (n_samples, 2).
    output_dir : Path
        Directory to save the HTML plots.
    save_html : bool, optional
        Whether to save the figures as HTML files. If False, returns the figures only.
    template : str, optional
        Plotly template style (e.g., 'plotly_white', 'seaborn', 'simple_white').

    Returns
    -------
    Dict[str, Path or go.Figure]
        Mapping of plot names to saved file paths or Plotly figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Validation ---
    if "cluster" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column for visualization")
    if pca_2d.shape[1] != 2:
        raise ValueError("pca_2d must be of shape (n_samples, 2)")
    if len(df) != len(pca_2d):
        raise ValueError("df and pca_2d must have the same number of rows")

    plot_results: Dict[str, Optional[Path]] = {}

<<<<<<< HEAD
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
	if "year" in df.columns:
		trend_df = (
			df.assign(year=pd.to_numeric(df["year"], errors="coerce"))
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
=======
    # --- 1️⃣ PCA Scatter Plot with Cluster Centroids ---
    scatter_df = df.copy()
    scatter_df["pca_x"], scatter_df["pca_y"] = pca_2d[:, 0], pca_2d[:, 1]

    centroids = scatter_df.groupby("cluster")[["pca_x", "pca_y"]].mean().reset_index()
>>>>>>> 65ec2bd (Updated Visualization.py)

    fig_scatter = px.scatter(
        scatter_df,
        x="pca_x",
        y="pca_y",
        color=scatter_df["cluster"].astype(str),
        hover_data={
            "title": True,
            "year": True,
            "cluster": True,
            "pca_x": ':.2f',
            "pca_y": ':.2f'
        },
        title="Clusters in 2D PCA Space",
        template=template,
    )

    # Add centroid markers
    fig_scatter.add_trace(
        go.Scatter(
            x=centroids["pca_x"],
            y=centroids["pca_y"],
            mode="markers+text",
            text=[f"C{c}" for c in centroids["cluster"]],
            textposition="top center",
            marker=dict(size=12, symbol="x", color="black"),
            name="Centroids",
        )
    )

    fig_scatter.update_layout(
        legend_title_text="Cluster",
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=40),
        width=900,
        height=700,
    )

    if save_html:
        scatter_path = output_dir / "clusters_scatter.html"
        fig_scatter.write_html(str(scatter_path), include_plotlyjs="cdn")
        plot_results["scatter"] = scatter_path
        logging.info("✅ Saved PCA scatter plot to %s", scatter_path)
    else:
        plot_results["scatter"] = fig_scatter

    # --- 2️⃣ Yearly Trend Plot ---
    if "year" not in df.columns:
        logging.warning("No 'year' column found — skipping yearly trend plot.")
        return plot_results

    trend_df = (
        df.assign(year=df["year"].astype(int))
        .groupby(["year", "cluster"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    fig_trend = px.line(
        trend_df,
        x="year",
        y="count",
        color=trend_df["cluster"].astype(str),
        title="Yearly Trend by Cluster",
        markers=True,
        template=template,
    )

    fig_trend.update_traces(line=dict(width=2))
    fig_trend.update_layout(
        legend_title_text="Cluster",
        title_x=0.5,
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if save_html:
        trend_path = output_dir / "yearly_trend.html"
        fig_trend.write_html(str(trend_path), include_plotlyjs="cdn")
        plot_results["trend"] = trend_path
        logging.info("✅ Saved yearly trend plot to %s", trend_path)
    else:
        plot_results["trend"] = fig_trend

    return plot_results
