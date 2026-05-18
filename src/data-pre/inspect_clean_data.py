import os
import re
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = r"D:\dl\dl\data\processed\legal_documents_clean_2024_2026.csv"
REPORT_DIR = r"D:\dl\dl\data\processed\reports_clean_data"


# ============================================================
# Regex
# ============================================================

ARTICLE_RE = re.compile(r"(?m)^\s*Điều\s+\d+[a-zA-Z]?\.")
CLAUSE_RE = re.compile(r"(?m)^\s*\d+\.\s+")
POINT_RE = re.compile(r"(?m)^\s*[a-zA-ZđĐ][\)\.]\s+")

APPENDIX_RE = re.compile(r"(?im)^\s*PHỤ\s+LỤC\s+([IVXLCDM\d]+)?\s*$")
SIGNATURE_RE = re.compile(r"(?im)^\s*Nơi nhận\s*:|^\s*TM\.\s+|^\s*KT\.\s+|^\s*CHỦ TỊCH\s*$")
TABLE_LINE_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")


# ============================================================
# Helpers
# ============================================================

def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def count_regex(text, pattern: re.Pattern) -> int:
    text = safe_text(text)
    return len(pattern.findall(text))


def has_regex(text, pattern: re.Pattern) -> bool:
    text = safe_text(text)
    return bool(pattern.search(text))


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_sample(df: pd.DataFrame, path: str, n: int = 50):
    if len(df) == 0:
        print(f"Không có dữ liệu để lưu: {path}")
        return
    df.head(n).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


def truncate_text(text, max_len=1000):
    text = safe_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...[TRUNCATED]..."


# ============================================================
# Main inspection
# ============================================================

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print_section("LOAD CLEAN DATA")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Input: {INPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    required_cols = [
        "id",
        "document_number",
        "title",
        "url",
        "legal_type",
        "legal_sectors",
        "issuing_authority",
        "issuance_date",
        "signers",
        "content_clean",
        "context_search",
        "raw_char_len",
        "clean_char_len",
        "had_appendix",
        "had_signature_block",
        "has_articles",
    ]

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        print("\n⚠️ Missing required columns:")
        print(missing_required)
    else:
        print("\n✅ Required columns đầy đủ.")

    # Normalize types
    df["content_clean"] = df["content_clean"].fillna("").astype(str)
    df["context_search"] = df["context_search"].fillna("").astype(str)

    if "clean_char_len" not in df.columns:
        df["clean_char_len"] = df["content_clean"].str.len()

    if "raw_char_len" not in df.columns:
        df["raw_char_len"] = df["content_clean"].str.len()

    # Recompute structural signals
    df["article_count_check"] = df["content_clean"].apply(lambda x: count_regex(x, ARTICLE_RE))
    df["clause_count_check"] = df["content_clean"].apply(lambda x: count_regex(x, CLAUSE_RE))
    df["point_count_check"] = df["content_clean"].apply(lambda x: count_regex(x, POINT_RE))

    df["has_articles_check"] = df["article_count_check"] > 0
    df["still_has_appendix_check"] = df["content_clean"].apply(lambda x: has_regex(x, APPENDIX_RE))
    df["still_has_signature_check"] = df["content_clean"].apply(lambda x: has_regex(x, SIGNATURE_RE))
    df["still_has_table_lines_check"] = df["content_clean"].apply(lambda x: has_regex(x, TABLE_LINE_RE))

    # ========================================================
    # Basic stats
    # ========================================================

    print_section("BASIC STATS")

    print(f"Tổng số documents: {len(df):,}")
    print(f"Duplicate id: {df['id'].duplicated().sum():,}" if "id" in df.columns else "Không có cột id")

    if "has_articles" in df.columns:
        print("\nhas_articles từ file:")
        print(df["has_articles"].value_counts(dropna=False))

    print("\nhas_articles recompute:")
    print(df["has_articles_check"].value_counts(dropna=False))

    if "has_articles" in df.columns:
        mismatch = df[df["has_articles"].astype(str).str.lower() != df["has_articles_check"].astype(str).str.lower()]
        print(f"\nMismatch has_articles vs recompute: {len(mismatch):,}")
        if len(mismatch) > 0:
            mismatch_path = os.path.join(REPORT_DIR, "mismatch_has_articles.csv")
            mismatch[
                [
                    "id",
                    "document_number",
                    "legal_type",
                    "title",
                    "has_articles",
                    "has_articles_check",
                    "article_count_check",
                    "clean_char_len",
                ]
            ].head(200).to_csv(mismatch_path, index=False, encoding="utf-8-sig")
            print(f"Saved mismatch sample: {mismatch_path}")

    # ========================================================
    # Missing metadata
    # ========================================================

    print_section("MISSING METADATA")

    metadata_cols = [
        "document_number",
        "title",
        "url",
        "legal_type",
        "legal_sectors",
        "issuing_authority",
        "issuance_date",
        "signers",
    ]

    for col in metadata_cols:
        if col in df.columns:
            missing = df[col].isna().sum() + (df[col].fillna("").astype(str).str.strip() == "").sum()
            print(f"{col:20s}: {missing:,}")

    missing_meta_cols = [c for c in metadata_cols if c in df.columns]
    if missing_meta_cols:
        df["missing_metadata_count"] = 0
        for col in missing_meta_cols:
            df["missing_metadata_count"] += df[col].fillna("").astype(str).str.strip().eq("").astype(int)

        missing_meta_df = df[df["missing_metadata_count"] > 0].copy()
        print(f"\nRows thiếu ít nhất 1 metadata field: {len(missing_meta_df):,}")

        if len(missing_meta_df) > 0:
            save_sample(
                missing_meta_df[
                    [
                        "id",
                        "document_number",
                        "title",
                        "legal_type",
                        "issuing_authority",
                        "issuance_date",
                        "missing_metadata_count",
                    ]
                ],
                os.path.join(REPORT_DIR, "missing_metadata_sample.csv"),
                n=200,
            )

    # ========================================================
    # Legal type distribution
    # ========================================================

    print_section("LEGAL TYPE DISTRIBUTION")

    if "legal_type" in df.columns:
        legal_type_counts = df["legal_type"].fillna("").replace("", "UNKNOWN").value_counts()
        print(legal_type_counts.head(30))

        legal_type_counts.to_csv(
            os.path.join(REPORT_DIR, "legal_type_distribution.csv"),
            encoding="utf-8-sig",
        )

    print_section("LEGAL TYPE DISTRIBUTION BY has_articles")

    if "legal_type" in df.columns:
        pivot = pd.crosstab(
            df["legal_type"].fillna("").replace("", "UNKNOWN"),
            df["has_articles_check"],
            margins=True,
        )
        print(pivot.sort_values(by="All", ascending=False).head(40))

        pivot.to_csv(
            os.path.join(REPORT_DIR, "legal_type_by_has_articles.csv"),
            encoding="utf-8-sig",
        )

    # ========================================================
    # Length stats
    # ========================================================

    print_section("LENGTH STATS")

    print(df["clean_char_len"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    length_stats = df["clean_char_len"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    length_stats.to_csv(
        os.path.join(REPORT_DIR, "clean_char_len_stats.csv"),
        encoding="utf-8-sig",
    )

    print("\nLength by has_articles:")
    print(df.groupby("has_articles_check")["clean_char_len"].describe())

    df.groupby("has_articles_check")["clean_char_len"].describe().to_csv(
        os.path.join(REPORT_DIR, "length_by_has_articles.csv"),
        encoding="utf-8-sig",
    )

    # ========================================================
    # Longest / shortest docs
    # ========================================================

    print_section("TOP LONGEST DOCUMENTS")

    top_long = df.sort_values("clean_char_len", ascending=False).head(50).copy()
    display_cols = [
        "id",
        "document_number",
        "legal_type",
        "title",
        "issuing_authority",
        "issuance_date",
        "clean_char_len",
        "article_count_check",
        "clause_count_check",
        "point_count_check",
        "has_articles_check",
    ]
    display_cols = [c for c in display_cols if c in top_long.columns]

    print(top_long[display_cols].head(20).to_string())

    top_long[display_cols + ["content_clean"]].to_csv(
        os.path.join(REPORT_DIR, "top_longest_documents.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print_section("TOO SHORT DOCUMENTS")

    short_docs = df[df["clean_char_len"] <= 300].copy()
    print(f"Documents <= 300 chars: {len(short_docs):,}")

    if len(short_docs) > 0:
        save_sample(
            short_docs[
                [
                    "id",
                    "document_number",
                    "legal_type",
                    "title",
                    "clean_char_len",
                    "content_clean",
                ]
            ],
            os.path.join(REPORT_DIR, "too_short_documents.csv"),
            n=200,
        )

    # ========================================================
    # Residual appendix/signature/table
    # ========================================================

    print_section("RESIDUAL APPENDIX / SIGNATURE / TABLE CHECK")

    print(f"Still has PHỤ LỤC: {df['still_has_appendix_check'].sum():,}")
    print(f"Still has signature block: {df['still_has_signature_check'].sum():,}")
    print(f"Still has table lines: {df['still_has_table_lines_check'].sum():,}")

    residual_cols = [
        "id",
        "document_number",
        "legal_type",
        "title",
        "clean_char_len",
        "still_has_appendix_check",
        "still_has_signature_check",
        "still_has_table_lines_check",
        "content_clean",
    ]

    residual_appendix = df[df["still_has_appendix_check"]].copy()
    if len(residual_appendix) > 0:
        save_sample(
            residual_appendix[residual_cols],
            os.path.join(REPORT_DIR, "residual_appendix_sample.csv"),
            n=100,
        )

    residual_signature = df[df["still_has_signature_check"]].copy()
    if len(residual_signature) > 0:
        save_sample(
            residual_signature[residual_cols],
            os.path.join(REPORT_DIR, "residual_signature_sample.csv"),
            n=100,
        )

    residual_table = df[df["still_has_table_lines_check"]].copy()
    if len(residual_table) > 0:
        save_sample(
            residual_table[residual_cols],
            os.path.join(REPORT_DIR, "residual_table_sample.csv"),
            n=100,
        )

    # ========================================================
    # Article structure stats
    # ========================================================

    print_section("ARTICLE / CLAUSE / POINT STRUCTURE")

    print("Article count:")
    print(df["article_count_check"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nClause count:")
    print(df["clause_count_check"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nPoint count:")
    print(df["point_count_check"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    structure_stats = df[
        [
            "article_count_check",
            "clause_count_check",
            "point_count_check",
            "clean_char_len",
        ]
    ].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    structure_stats.to_csv(
        os.path.join(REPORT_DIR, "structure_stats.csv"),
        encoding="utf-8-sig",
    )

    # ========================================================
    # Samples for manual inspection
    # ========================================================

    print_section("SAVE MANUAL INSPECTION SAMPLES")

    has_articles_df = df[df["has_articles_check"]].copy()
    no_articles_df = df[~df["has_articles_check"]].copy()

    sample_cols = [
        "id",
        "document_number",
        "title",
        "legal_type",
        "legal_sectors",
        "issuing_authority",
        "issuance_date",
        "clean_char_len",
        "article_count_check",
        "clause_count_check",
        "point_count_check",
        "content_clean",
    ]
    sample_cols = [c for c in sample_cols if c in df.columns]

    save_sample(
        has_articles_df[sample_cols].sample(
            min(100, len(has_articles_df)),
            random_state=42,
        ),
        os.path.join(REPORT_DIR, "sample_has_articles.csv"),
        n=100,
    )

    save_sample(
        no_articles_df[sample_cols].sample(
            min(100, len(no_articles_df)),
            random_state=42,
        ),
        os.path.join(REPORT_DIR, "sample_no_articles.csv"),
        n=100,
    )

    # Console preview
    print("\nPreview has_articles=True:")
    for _, row in has_articles_df.head(3).iterrows():
        print("-" * 100)
        print(row.get("id"), row.get("document_number"), row.get("title"))
        print(truncate_text(row.get("content_clean"), 1200))

    print("\nPreview has_articles=False:")
    for _, row in no_articles_df.head(3).iterrows():
        print("-" * 100)
        print(row.get("id"), row.get("document_number"), row.get("title"))
        print(truncate_text(row.get("content_clean"), 1200))

    # ========================================================
    # Final recommendation
    # ========================================================

    print_section("RECOMMENDATION")

    total = len(df)
    has_article_count = int(df["has_articles_check"].sum())
    no_article_count = total - has_article_count
    residual_appendix_count = int(df["still_has_appendix_check"].sum())
    residual_signature_count = int(df["still_has_signature_check"].sum())
    residual_table_count = int(df["still_has_table_lines_check"].sum())

    print(f"Total docs: {total:,}")
    print(f"Has articles: {has_article_count:,} ({has_article_count / total:.2%})")
    print(f"No articles: {no_article_count:,} ({no_article_count / total:.2%})")
    print(f"Residual appendix: {residual_appendix_count:,}")
    print(f"Residual signature: {residual_signature_count:,}")
    print(f"Residual table lines: {residual_table_count:,}")

    if residual_appendix_count > 0:
        print("\n⚠️ Vẫn còn văn bản chứa PHỤ LỤC. Nên xem reports_clean_data/residual_appendix_sample.csv")

    if residual_signature_count > 0:
        print("\n⚠️ Vẫn còn văn bản chứa Nơi nhận/TM./KT./CHỦ TỊCH. Nên xem residual_signature_sample.csv")

    if residual_table_count > 0:
        print("\n⚠️ Vẫn còn dòng bảng markdown. Nên xem residual_table_sample.csv")

    print("\nChiến lược chunk đề xuất:")
    print("- has_articles=True  -> chunk theo Điều/Khoản/Điểm.")
    print("- has_articles=False -> chunk fallback theo heading/paragraph/sliding window.")
    print("- Không bỏ has_articles=False, vì vẫn là dữ liệu có ích.")
    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()