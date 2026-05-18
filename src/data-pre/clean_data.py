import os
import re
import unicodedata
import pandas as pd
from datasets import load_from_disk


METADATA_PATH = r"D:\dl\dl\data\raw\vietnamese_legal_metadata"
CONTENT_PATH = r"D:\dl\dl\data\raw\vietnamese_legal_content"
OUTPUT_PATH = r"D:\dl\dl\data\processed\legal_documents_clean_2026.csv"


WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

APPENDIX_RE = re.compile(
    r"(?im)^\s*PHỤ\s+LỤC\s+([IVXLCDM\d]+)?\s*$"
)

SIGNATURE_RE = re.compile(
    r"(?im)^\s*Nơi nhận\s*:|^\s*TM\.\s+|^\s*KT\.\s+|^\s*CHỦ TỊCH\s*$"
)

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def clean_text_basic(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")

    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def remove_appendix_and_after(text):
    text = clean_text_basic(text)
    match = APPENDIX_RE.search(text)
    if match:
        return text[:match.start()].strip()
    return text


def remove_signature_block(text):
    text = clean_text_basic(text)
    match = SIGNATURE_RE.search(text)
    if match:
        return text[:match.start()].strip()
    return text


def remove_markdown_table_lines(text):
    lines = clean_text_basic(text).splitlines()
    cleaned_lines = []

    for line in lines:
        if TABLE_LINE_RE.match(line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def clean_legal_content(text):
    text = clean_text_basic(text)
    text = remove_appendix_and_after(text)
    text = remove_signature_block(text)
    text = remove_markdown_table_lines(text)
    text = clean_text_basic(text)
    return text


def make_context_search(text):
    text = clean_text_basic(text).lower()
    text = re.sub(r"[^\w\s/\.\-:;,%()\[\]đĐ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_articles(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"(?m)^\s*Điều\s+\d+[a-zA-Z]?\.", text))


def has_appendix(text):
    if not isinstance(text, str):
        return False
    return bool(APPENDIX_RE.search(text))


def has_signature(text):
    if not isinstance(text, str):
        return False
    return bool(SIGNATURE_RE.search(text))


def main():
    print("📥 Loading datasets from disk...")

    metadata = load_from_disk(METADATA_PATH)
    content = load_from_disk(CONTENT_PATH)

    metadata_df = metadata["data"].to_pandas()
    content_df = content["data"].to_pandas()

    print(f"Metadata rows: {len(metadata_df):,}")
    print(f"Content rows:  {len(content_df):,}")

    print("🔗 Merging content + metadata...")
    df = content_df.merge(metadata_df, on="id", how="left")

    print("Missing metadata rows:", df["document_number"].isna().sum())

    df["issuance_date_parsed"] = pd.to_datetime(
        df["issuance_date"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    df = df[
        (df["issuance_date_parsed"].dt.year >= 2026)
        & (df["issuance_date_parsed"].dt.year <= 2026)
    ].copy()

    print(f"Rows after 2024–2026 filter: {len(df):,}")

    keep_cols = [
        "id",
        "document_number",
        "title",
        "url",
        "legal_type",
        "legal_sectors",
        "issuing_authority",
        "issuance_date",
        "signers",
        "content",
    ]

    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""

    print("🧹 Cleaning content...")

    df["content_raw"] = df["content"].fillna("").astype(str)

    df["had_appendix"] = df["content_raw"].apply(has_appendix)
    df["had_signature_block"] = df["content_raw"].apply(has_signature)

    df["content_clean"] = df["content_raw"].apply(clean_legal_content)
    df["context_search"] = df["content_clean"].apply(make_context_search)

    df["raw_char_len"] = df["content_raw"].str.len()
    df["clean_char_len"] = df["content_clean"].str.len()
    df["has_articles"] = df["content_clean"].apply(has_articles)

    before = len(df)
    df = df[df["clean_char_len"] > 100].copy()
    after = len(df)

    print(f"Removed empty/too-short docs after clean: {before - after:,}")

    output_cols = [
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

    output_df = df[output_cols].copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== DONE =====")
    print(f"Tổng số bản ghi sau filter + clean: {len(output_df):,}")
    print(f"Có phụ lục bị xóa: {output_df['had_appendix'].sum():,}")
    print(f"Có chữ ký/nơi nhận bị xóa: {output_df['had_signature_block'].sum():,}")
    print(f"Có Điều sau clean: {output_df['has_articles'].sum():,}")
    print(f"Không có Điều sau clean: {(~output_df['has_articles']).sum():,}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSample:")
    print(
        output_df[
            [
                "id",
                "document_number",
                "legal_type",
                "issuing_authority",
                "issuance_date",
                "clean_char_len",
                "has_articles",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()