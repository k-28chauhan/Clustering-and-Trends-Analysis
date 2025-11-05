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
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input JSON or CSV file",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Names of the 4 fields/columns to use for clustering",
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

    # 1️⃣ Load data (JSON or CSV)
    if not args.fields:
        raise ValueError("Please specify the 4 fields using --fields field1 field2 field3 field4")

    df: pd.DataFrame = load_selected_fields(args.data, args.fields)
    logging.info(f"Loaded {len(df)} rows with fields: {args.fields}")

    # 2️⃣ Generate embeddings (auto-handles numeric/text) — optimized
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set path for cached embeddings (unique per model)
    cache_file = output_dir / f"embeddings_{args.model.replace('/', '_')}.npy"

    # 2️⃣ Identify which fields to use for clustering (exclude Year)
    cluster_fields = [f for f in args.fields if f.lower() != "year"]

    # 3️⃣ Generate embeddings (auto-handles numeric/text)
    embeddings = generate_embeddings(
        df,
        text_columns=cluster_fields,           # fields used for clustering
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,                         # GPU or CPU
        cache_path=cache_file,                 # cache path
        overwrite_cache=False,                 # reuse cache if exists
        verbose=(args.verbosity >= 1),         # show extra info in logs
    )


    # 3️⃣ KMeans clustering + PCA
    labels, pca_2d, _ = run_clustering_and_pca(
        embeddings,
        n_clusters=args.clusters,
    )

    # 4️⃣ Attach results & visualize
    df_out = df.copy()
    df_out["cluster"] = labels
    plot_paths = create_interactive_visualizations(df_out, pca_2d, output_dir)

    # 5️⃣ Save results
    csv_path = output_dir / "clustered_results.csv"
    df_out.to_csv(csv_path, index=False)
    logging.info(f"Saved clustered data to {csv_path}")

    print("\n✅ Outputs:")
    print(f"- Clustered CSV: {csv_path}")
    for name, path in plot_paths.items():
        print(f"- {name.capitalize()} plot: {path}")


if __name__ == "__main__":
    main()
