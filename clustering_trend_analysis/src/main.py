import json
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch  # moved inside imports for clarity

from .load_data import load_selected_fields  # ✅ updated import
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
        description="Clustering and Visualization of Data (JSON or CSV)",
    )

    # --- Basic arguments ---
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input JSON or CSV file",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        required=True,
        help="Names of the fields/columns to use for clustering",
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
        default=None,
        help="Number of KMeans clusters. If not specified, defaults to 5.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/specter",
        help="SentenceTransformer model name (default: allenai/specter for academic papers)",
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

    # --- New optional flags ---
    parser.add_argument(
        "--no-sentence-pooling",
        action="store_true",
        help="Disable per-sentence pooling in embedding generation (default: enabled)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help='JSON string mapping column names to weights, e.g. \'{"title":3,"abstract":2}\'',
    )
    parser.add_argument(
        "--reduce-to",
        type=int,
        default=None,
        help="Optionally reduce embedding dimension (e.g. 256). Uses TruncatedSVD.",
    )

    args = parser.parse_args()
    configure_logging(args.verbosity)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df: pd.DataFrame = load_selected_fields(args.data, args.fields)
    logging.info(f"Loaded {len(df)} rows with fields: {args.fields}")

    # --- Device selection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Caching path ---
    cache_file = output_dir / f"embeddings_{args.model.replace('/', '_')}.npy"

    # --- Parse custom field weights ---
    field_weights = None
    if args.weights:
        try:
            field_weights = json.loads(args.weights)
            logging.info(f"Using custom field weights: {field_weights}")
        except json.JSONDecodeError:
            logging.warning("Invalid JSON for --weights. Ignoring custom weights.")

    # --- Generate embeddings ---
    embeddings = generate_embeddings(
        df,
        text_columns=args.fields,
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,
        cache_path=cache_file,
        overwrite_cache=False,
        verbose=(args.verbosity >= 1),
        field_weights=field_weights,
        per_sentence_pooling=not args.no_sentence_pooling,
        reduce_to=args.reduce_to,
    )

    # --- KMeans clustering + PCA ---
    labels, pca_2d, _ = run_clustering_and_pca(
        embeddings,
        n_clusters=args.clusters or 5,
    )

    # --- Visualization + save ---
    df_out = df.copy()
    df_out["cluster"] = labels
    plot_paths = create_interactive_visualizations(df_out, pca_2d, output_dir)

    csv_path = output_dir / "clustered_results.csv"
    df_out.to_csv(csv_path, index=False)
    logging.info(f"Saved clustered data to {csv_path}")

    print("\n✅ Outputs:")
    print(f"- Clustered CSV: {csv_path}")
    for name, path in plot_paths.items():
        print(f"- {name.capitalize()} plot: {path}")


if __name__ == "__main__":
    main()
