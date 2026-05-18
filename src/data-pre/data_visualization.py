import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_PATH = ROOT_DIR / "data" / "processed" / "legal_documents_clean_2024_2026.csv"
DEFAULT_CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "legal_chunks_2024_2026.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "data_visualization"


def require_plotting_libraries() -> None:
    if plt is None or sns is None:
        raise ImportError("Missing plotting libraries. Install them with: pip install matplotlib seaborn")


def set_plot_style() -> None:
    require_plotting_libraries()
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "issuance_date" not in df.columns:
        return df

    parsed = pd.to_datetime(df["issuance_date"], format="%d/%m/%Y", errors="coerce")
    df["issuance_date_parsed"] = parsed
    df["issuance_year"] = parsed.dt.year.astype("Int64")
    df["issuance_month"] = parsed.dt.to_period("M").astype(str)
    df.loc[parsed.isna(), "issuance_month"] = "UNKNOWN"
    return df


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=bool)
    return df[column].astype(str).str.lower().isin(["true", "1", "yes"])


def top_counts(df: pd.DataFrame, column: str, top_n: int) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count", "percent"])

    counts = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "UNKNOWN")
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )
    counts["percent"] = (counts["count"] / len(df) * 100).round(2) if len(df) else 0
    return counts


def save_summary_csv(table: pd.DataFrame, output_dir: Path, name: str) -> None:
    table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def save_horizontal_bar(
    table: pd.DataFrame,
    label_col: str,
    title: str,
    output_path: Path,
    color: str,
) -> Optional[Path]:
    if table.empty or label_col not in table.columns or "count" not in table.columns:
        return None

    plot_df = table.copy()
    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df["count"] = pd.to_numeric(plot_df["count"], errors="coerce").fillna(0)
    plot_df = plot_df.sort_values("count", ascending=True)

    height = max(5.5, min(11.5, 0.45 * len(plot_df) + 2))
    fig, ax = plt.subplots(figsize=(12, height))
    sns.barplot(data=plot_df, x="count", y=label_col, ax=ax, color=color)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=4, fontsize=9)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_vertical_bar(
    table: pd.DataFrame,
    label_col: str,
    title: str,
    output_path: Path,
    color: str,
) -> Optional[Path]:
    if table.empty or label_col not in table.columns or "count" not in table.columns:
        return None

    plot_df = table.copy()
    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df["count"] = pd.to_numeric(plot_df["count"], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x=label_col, y="count", ax=ax, color=color)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=3, fontsize=9)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
    color: str,
    bins: int = 50,
    xmax_quantile: Optional[float] = None,
) -> Optional[Path]:
    if column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None

    if xmax_quantile is not None:
        upper = values.quantile(xmax_quantile)
        values = values[values <= upper]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(values, bins=bins, ax=ax, color=color)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_pie_chart(
    values: pd.Series,
    title: str,
    output_path: Path,
    labels: List[str],
    colors: List[str],
) -> Optional[Path]:
    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 10},
    )
    ax.set_title(title, fontsize=16, pad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_overview_grid(
    documents: pd.DataFrame,
    chunks: Optional[pd.DataFrame],
    output_path: Path,
) -> Path:
    metrics = [
        ("Documents", len(documents)),
        ("Chunks", len(chunks) if chunks is not None else 0),
        ("Docs with articles", int(bool_series(documents, "has_articles").sum())),
        ("Docs with appendix removed", int(bool_series(documents, "had_appendix").sum())),
        ("Docs with signature removed", int(bool_series(documents, "had_signature_block").sum())),
    ]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    cell_text = [[label, f"{value:,}"] for label, value in metrics]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    ax.set_title("Legal Dataset Overview", fontsize=18, pad=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_document_charts(documents: pd.DataFrame, output_dir: Path, top_n: int) -> List[Path]:
    charts: List[Path] = []

    type_counts = top_counts(documents, "legal_type", top_n)
    sector_counts = top_counts(documents, "legal_sectors", top_n)
    authority_counts = top_counts(documents, "issuing_authority", top_n)
    year_counts = top_counts(documents, "issuance_year", 20)

    save_summary_csv(type_counts, output_dir, "top_document_types")
    save_summary_csv(sector_counts, output_dir, "top_legal_sectors")
    save_summary_csv(authority_counts, output_dir, "top_issuing_authorities")
    save_summary_csv(year_counts, output_dir, "documents_by_year")

    chart_specs = [
        save_horizontal_bar(
            type_counts,
            "legal_type",
            f"Top {top_n} Document Types",
            output_dir / f"top_{top_n}_document_types.png",
            "#2563eb",
        ),
        save_horizontal_bar(
            sector_counts,
            "legal_sectors",
            f"Top {top_n} Legal Sectors",
            output_dir / f"top_{top_n}_legal_sectors.png",
            "#16a34a",
        ),
        save_horizontal_bar(
            authority_counts,
            "issuing_authority",
            f"Top {top_n} Issuing Authorities",
            output_dir / f"top_{top_n}_issuing_authorities.png",
            "#9333ea",
        ),
        save_vertical_bar(
            year_counts,
            "issuance_year",
            "Documents by Year",
            output_dir / "documents_by_year.png",
            "#ea580c",
        ),
        save_histogram(
            documents,
            "clean_char_len",
            "Document Length Distribution (<= p99)",
            output_dir / "document_length_distribution.png",
            "#0891b2",
            bins=50,
            xmax_quantile=0.99,
        ),
    ]

    if "has_articles" in documents.columns:
        article_values = bool_series(documents, "has_articles").value_counts().reindex([True, False], fill_value=0)
        path = save_pie_chart(
            article_values,
            "Documents With Article Structure",
            output_dir / "has_articles_distribution.png",
            labels=["Has articles", "No articles"],
            colors=["#2563eb", "#d1d5db"],
        )
        chart_specs.append(path)

    charts.extend(path for path in chart_specs if path is not None)
    return charts


def build_chunk_charts(chunks: Optional[pd.DataFrame], output_dir: Path, top_n: int) -> List[Path]:
    if chunks is None:
        return []

    charts: List[Path] = []
    chunk_type_counts = top_counts(chunks, "chunk_type", top_n)
    save_summary_csv(chunk_type_counts, output_dir, "top_chunk_types")

    chart_specs = [
        save_horizontal_bar(
            chunk_type_counts,
            "chunk_type",
            f"Top {top_n} Chunk Types",
            output_dir / "top_chunk_types.png",
            "#0f766e",
        ),
        save_histogram(
            chunks,
            "chunk_char_len",
            "Chunk Length Distribution (<= p99)",
            output_dir / "chunk_length_distribution.png",
            "#be123c",
            bins=50,
            xmax_quantile=0.99,
        ),
    ]

    if "doc_id" in chunks.columns:
        chunks_per_doc = chunks.groupby("doc_id").size().reset_index(name="chunks_per_doc")
        save_summary_csv(
            chunks_per_doc.sort_values("chunks_per_doc", ascending=False).head(50),
            output_dir,
            "top_docs_by_chunk_count",
        )
        chart_specs.append(
            save_histogram(
                chunks_per_doc,
                "chunks_per_doc",
                "Chunks per Document Distribution (<= p99)",
                output_dir / "chunks_per_document_distribution.png",
                "#7c3aed",
                bins=40,
                xmax_quantile=0.99,
            )
        )

    charts.extend(path for path in chart_specs if path is not None)
    return charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create seaborn/matplotlib charts for legal clean/chunk data.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH), help="Clean documents CSV.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), help="Chunks CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for PNG charts and CSV summaries.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top categories to plot.")
    parser.add_argument("--skip-chunks", action="store_true", help="Only plot document-level charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    documents_path = Path(args.documents)
    chunks_path = Path(args.chunks)
    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"
    summaries_dir = output_dir / "summaries"
    charts_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    documents = read_csv(documents_path)
    documents = ensure_numeric(documents, ["raw_char_len", "clean_char_len"])
    documents = add_date_columns(documents)

    chunks = None
    if not args.skip_chunks:
        if chunks_path.exists():
            chunks = read_csv(chunks_path)
            chunks = ensure_numeric(chunks, ["chunk_char_len", "chunk_index"])
        else:
            print(f"Chunks CSV not found, skipping chunk charts: {chunks_path}")

    charts = [save_overview_grid(documents, chunks, charts_dir / "dataset_overview.png")]
    charts.extend(build_document_charts(documents, charts_dir, args.top_n))
    charts.extend(build_chunk_charts(chunks, charts_dir, args.top_n))

    # Keep compact CSV summaries beside the charts for quick value checks.
    for csv_path in charts_dir.glob("*.csv"):
        csv_path.replace(summaries_dir / csv_path.name)

    print("\n===== DONE =====")
    print(f"Documents: {len(documents):,}")
    if chunks is not None:
        print(f"Chunks:    {len(chunks):,}")
    print(f"Charts saved to: {charts_dir}")
    print(f"Summaries saved to: {summaries_dir}")
    for path in charts:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
