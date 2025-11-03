import argparse
import logging
from pathlib import Path

import pandas as pd

from .load_data import load_json_to_df
from .embeddings import generate_embeddings
from .clustering import run_clustering_and_pca
from .visualization import create_interactive_visualizations


def configure_logging(verbosity: int) -> None:
	level = logging.WARNING
	if verbosity == 1:
		level = logging.INFO
	elif verbosity >= 2:
		level = logging.DEBUG
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Clustering and Identification of Trends in Summarised PDF Data",
	)
	parser.add_argument(
		"--data",
		type=Path,
		default=Path(__file__).resolve().parent.parent / "data" / "sample_data.json",
		help="Path to input JSON file",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent.parent / "outputs",
		help="Directory to save outputs",
	)
	parser.add_argument(
		"--clusters",
		type=int,
		default=3,
		help="Number of KMeans clusters",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="sentence-transformers/all-MiniLM-L6-v2",
		help="SentenceTransformer model name",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size for embedding generation",
	)
	parser.add_argument(
		"--verbosity",
		type=int,
		choices=[0, 1, 2],
		default=1,
		help="0=warn, 1=info, 2=debug",
	)
	args = parser.parse_args()

	configure_logging(args.verbosity)

	output_dir: Path = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1. Load JSON → DataFrame
	df: pd.DataFrame = load_json_to_df(args.data)

	# 2-3. Combine text fields and generate embeddings
	text_columns = ["problem", "findings", "limitations"]
	embeddings = generate_embeddings(
		df,
		text_columns=text_columns,
		model_name=args.model,
		batch_size=args.batch_size,
	)

	# 4-5. KMeans clustering and PCA 2D reduction
	labels, pca_2d, _ = run_clustering_and_pca(embeddings, n_clusters=args.clusters)

	# 6. Attach results and visualize
	df_out = df.copy()
	df_out["cluster"] = labels

	plot_paths = create_interactive_visualizations(df_out, pca_2d, output_dir)

	# 7. Save clustered results
	csv_path = output_dir / "clustered_results.csv"
	df_out.to_csv(csv_path, index=False)
	logging.info("Saved clustered data to %s", csv_path)

	print("Outputs:")
	print(f"- Clustered CSV: {csv_path}")
	for name, path in plot_paths.items():
		print(f"- {name.capitalize()} plot: {path}")


if __name__ == "__main__":
	main()


