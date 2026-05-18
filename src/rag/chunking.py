import argparse
import os
import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = str(ROOT_DIR / "configs" / "rag" / "chunking.yaml")

ARTICLE_RE = re.compile("")
ARTICLE_TITLE_RE = re.compile("")
WHITESPACE_RE = re.compile("")
MULTI_NEWLINE_RE = re.compile("")
SEARCH_CLEAN_RE = re.compile("")
METADATA_COLUMNS: List[str] = []


def read_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("Chưa cài PyYAML. Hãy chạy: pip install PyYAML")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config YAML phải là object: {path}")
    return data


def apply_chunk_settings(settings: Dict[str, Any]) -> None:
    global ARTICLE_RE, ARTICLE_TITLE_RE, WHITESPACE_RE, MULTI_NEWLINE_RE, SEARCH_CLEAN_RE, METADATA_COLUMNS

    patterns = settings.get("patterns", {})
    columns = settings.get("columns", {})

    ARTICLE_RE = re.compile(patterns["article"])
    ARTICLE_TITLE_RE = re.compile(patterns["article_title"])
    WHITESPACE_RE = re.compile(patterns["whitespace"])
    MULTI_NEWLINE_RE = re.compile(patterns["multi_newline"])
    SEARCH_CLEAN_RE = re.compile(patterns["search_clean"], flags=re.UNICODE)
    METADATA_COLUMNS = list(columns.get("metadata", []))


def validate_chunk_config(config: SimpleNamespace) -> None:
    if config.max_chars < 1000:
        raise ValueError("max_chars nên >= 1000 để tránh chunk quá ngắn.")
    if config.overlap_chars < 0:
        raise ValueError("overlap_chars không được âm.")
    if config.overlap_chars >= config.max_chars:
        raise ValueError("overlap_chars phải nhỏ hơn max_chars.")
    if not config.text_col:
        raise ValueError("Config thiếu columns.text.")


def build_runtime_config(settings: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str, SimpleNamespace]:
    paths = settings.get("paths", {})
    columns = settings.get("columns", {})
    chunking = settings.get("chunking", {})

    input_path = args.input or paths.get("input", "")
    output_path = args.output or paths.get("output", "")
    config = SimpleNamespace(
        text_col=args.text_col or columns.get("text", ""),
        max_chars=args.max_chars if args.max_chars is not None else int(chunking.get("max_chars", 0)),
        overlap_chars=(
            args.overlap_chars
            if args.overlap_chars is not None
            else int(chunking.get("overlap_chars", 0))
        ),
    )

    if not input_path:
        raise ValueError("Config thiếu paths.input.")
    if not output_path:
        raise ValueError("Config thiếu paths.output.")
    validate_chunk_config(config)
    return input_path, output_path, config


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_for_search(text: str) -> str:
    text = clean_text(text).lower()
    text = SEARCH_CLEAN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_metadata_prefix(row: pd.Series) -> str:
    fields = [
        ("Tên văn bản", row.get("title", "")),
        ("Số hiệu", row.get("document_number", "")),
        ("Loại văn bản", row.get("legal_type", "")),
        ("Lĩnh vực", row.get("legal_sectors", "")),
        ("Cơ quan ban hành", row.get("issuing_authority", "")),
        ("Ngày ban hành", row.get("issuance_date", "")),
    ]
    lines = [f"{name}: {to_text(value)}" for name, value in fields if to_text(value)]
    return "\n".join(lines)


def parse_article_header(header: str) -> Tuple[str, str]:
    match = ARTICLE_TITLE_RE.match(header.strip())
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def split_legal_units(text: str) -> List[Dict[str, str]]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return []

    units: List[Dict[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        units.append({
            "type": "preamble",
            "article_number": "",
            "article_title": "",
            "text": preamble,
        })

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        article_number, article_title = parse_article_header(match.group(1))
        units.append({
            "type": "article",
            "article_number": article_number,
            "article_title": article_title,
            "text": block,
        })

    return units


def split_fixed(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            cut = text.rfind(" ", start + int(max_chars * 0.7), end)
            if cut > start:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = max(0, end - overlap_chars)
        start = end if next_start <= start else next_start

    return chunks


def split_large_block(block: str, max_chars: int, overlap_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?;:])\s+", block)
    if len(sentences) == 1:
        return split_fixed(block, max_chars, overlap_chars)
    return pack_blocks(sentences, max_chars, overlap_chars)


def overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return ""

    tail = text[-overlap_chars:].strip()
    cut_positions = [tail.find(". "), tail.find("\n")]
    cut_positions = [pos for pos in cut_positions if pos >= 0]
    if cut_positions:
        tail = tail[min(cut_positions) + 1 :].strip()
    return tail


def pack_blocks(blocks: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    chunks: List[str] = []
    current = ""

    for block in blocks:
        block = clean_text(block)
        if not block:
            continue

        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_large_block(block, max_chars, overlap_chars))
            continue

        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current)
        tail = overlap_tail(current, overlap_chars)
        current = f"{tail}\n\n{block}" if tail else block

    if current:
        chunks.append(current)

    return chunks


def split_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    blocks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if len(blocks) == 1:
        blocks = [line.strip() for line in text.splitlines() if line.strip()]

    return pack_blocks(blocks, max_chars, overlap_chars)


def split_article_text(article_text: str, config: SimpleNamespace) -> List[str]:
    article_text = clean_text(article_text)
    if len(article_text) <= config.max_chars:
        return [article_text]

    header, _, body = article_text.partition("\n")
    body = body.strip()
    if not body:
        return split_text(article_text, config.max_chars, config.overlap_chars)

    body_max_chars = max(1000, config.max_chars - len(header) - 2)
    body_chunks = split_text(body, body_max_chars, config.overlap_chars)
    return [f"{header}\n{chunk}".strip() for chunk in body_chunks]


def chunk_document(row: pd.Series, config: SimpleNamespace) -> List[Dict[str, Any]]:
    text = clean_text(row.get(config.text_col, ""))
    if not text:
        return []

    prefix = make_metadata_prefix(row)
    units = split_legal_units(text)
    if not units:
        units = [{"type": "paragraph", "article_number": "", "article_title": "", "text": text}]

    rows: List[Dict[str, Any]] = []
    doc_id = to_text(row.get("id", ""))

    for unit in units:
        if unit["type"] == "article":
            parts = split_article_text(unit["text"], config)
            chunk_type = "article" if len(parts) == 1 else "article_part"
        else:
            parts = split_text(unit["text"], config.max_chars, config.overlap_chars)
            chunk_type = unit["type"]

        for part in parts:
            chunk_content = clean_text(part)
            chunk_text = f"{prefix}\n\n{chunk_content}".strip() if prefix else chunk_content
            rows.append({
                "doc_id": doc_id,
                "chunk_index": len(rows),
                "chunk_id": f"{doc_id}__{len(rows):04d}",
                "chunk_type": chunk_type,
                "article_number": unit["article_number"],
                "article_title": unit["article_title"],
                "chunk_char_len": len(chunk_content),
                "chunk_text": chunk_text,
                "context_search": clean_for_search(chunk_text),
            })

    return rows


def build_chunks(df: pd.DataFrame, config: SimpleNamespace) -> pd.DataFrame:
    chunk_rows: List[Dict[str, Any]] = []
    missing_metadata = [column for column in METADATA_COLUMNS if column not in df.columns]
    if missing_metadata:
        print(
            "WARNING: input CSV is missing configured metadata columns. "
            f"Chunks will be created without these fields: {missing_metadata}"
        )

    for processed_count, (_, row) in enumerate(df.iterrows(), start=1):
        metadata = {column: row.get(column, "") for column in METADATA_COLUMNS if column in df.columns}
        for chunk in chunk_document(row, config):
            chunk_rows.append({**metadata, **chunk})

        if processed_count % 5000 == 0:
            print(f"Processed {processed_count:,} documents | chunks: {len(chunk_rows):,}")

    return pd.DataFrame(chunk_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk Vietnamese legal documents for RAG.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config cho chunking.")
    parser.add_argument("--input", default=None, help="Override paths.input trong YAML.")
    parser.add_argument("--output", default=None, help="Override paths.output trong YAML.")
    parser.add_argument("--text-col", default=None, help="Override columns.text trong YAML.")
    parser.add_argument("--max-chars", type=int, default=None, help="Override chunking.max_chars trong YAML.")
    parser.add_argument("--overlap-chars", type=int, default=None, help="Override chunking.overlap_chars trong YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = read_yaml_config(args.config)
    apply_chunk_settings(settings)
    input_path, output_path, config = build_runtime_config(settings, args)

    df = pd.read_csv(input_path)
    if config.text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột nội dung: {config.text_col}")

    chunks = build_chunks(df, config)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    chunks.to_csv(output_path, index=False, encoding="utf-8-sig")

    docs = sum(bool(clean_text(value)) for value in df[config.text_col])
    avg_chunks = len(chunks) / docs if docs else 0
    print("\n===== DONE =====")
    print(f"Documents: {docs:,}")
    print(f"Chunks:    {len(chunks):,}")
    print(f"Avg/doc:   {avg_chunks:.2f}")
    print(f"Saved to:  {output_path}")


if __name__ == "__main__":
    main()
