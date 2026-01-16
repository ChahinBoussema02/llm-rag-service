# app/rag/ingest.py
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import frontmatter

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    version: str
    last_updated: str
    applies_to: List[str]
    section_path: List[str]     # e.g. ["Plans", "Pro"]
    text: str                   # chunk text
    source_file: str            # source file path
    start_line: int             # for traceability
    end_line: int 

HEADER_RE = re.compile(r"^#{1,6}\s+(.*)\s*$")

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _split_sections(md: str) -> List[Tuple[List[str], str, int, int]]:
    """
    Split markdown into (section_path, section_text, start_line, end_line).
    Uses markdown headings (# .. ######) to build a hierarchy.
    """
    lines = md.splitlines()
    sections: List[Tuple[List[str], List[str], int]] = []
    current_path: List[str] = ["(root)"]
    current_buf: List[str] = []
    current_start = 1

    def flush(end_line: int):
        nonlocal current_buf, current_start
        text = "\n".join(current_buf).strip()
        if text:
            sections.append((current_path.copy(), current_buf.copy(), current_start))

        current_buf = []
        current_start = end_line + 1

    #Track heading stack: list of (level, title)
    stack: List[Tuple[int, int]] = []

    for i, line in enumerate(lines, start = 1):
        m = HEADER_RE.match(line)
        if m:
            # Flush previous section
            flush(i - 1)

            # New heading
            title = _clean(m.group(1))
            level = len(line) - len(line.lstrip('#'))
            
           # Adjust stack
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            current_path = [t for _, t in stack]
            current_start = i + 1
        else:
            current_buf.append(line)
    
    #flush final
    flush(len(lines))

    #Convert to final tuples with line ranges
    out: List[Tuple[List[str], str, int, int]] = []
    for path, buf, start in sections:
        text = "\n".join(buf).strip()
        
        #estimate end line as start + number of lines - 1
        end = start + max(0, len(buf) - 1)
        out.append((path, text, start, end))
    return out

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Good enough for Phase 2; later you can switch to token-aware chunking.
    """

    text = text.strip()
    if not text:
        return []
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    
    return chunks

def ingest_markdown_dir(
    input_dir: Path,
    output_path: Path,
    max_chars: int = 900,
    overlap: int = 120,
) -> int:
    md_files = sorted(input_dir.glob("*.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for md_file in md_files:
            post = frontmatter.load(md_file)

            meta: Dict[str, object] = dict(post.metadata or {})
            body: str = post.content or ""

            #required metadata
            doc_id = str(meta.get("doc_id", md_file.stem))
            title = str(meta.get("title", md_file.stem))
            category = str(meta.get("category", "unknown"))
            version = str(meta.get("version", "1.0"))
            last_updated = str(meta.get("last_updated", ""))
            applies_to = meta.get("applies_to", [])
            if not isinstance(applies_to, list):
                applies_to = [str(applies_to)]
            applies_to = [str(a) for a in applies_to]

            # Split into sections
            sections = _split_sections(body)

            chunk_idx = 0
            for section_path, section_text, start_line, end_line in sections:
                #skip super tiny sections
                if len(section_text.strip()) < 40:
                    continue

                for piece in _chunk_text(section_text, max_chars, overlap):
                    chunk_id = f"{doc_id}::c{chunk_idx:04d}"
                    chunk_idx += 1
                    rec = Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=title,
                        category=category,
                        version=version,
                        last_updated=last_updated,
                        applies_to=applies_to,
                        section_path=section_path,
                        text=piece,
                        source_file=md_file.name,
                        start_line=start_line,
                        end_line=end_line,
                    )
                    out.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                    count += 1
    return count

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = repo_root / "data" / "raw" / "company_docs"
    output_path = repo_root / "data" / "processed" / "chunks.jsonl"

    n = ingest_markdown_dir(input_dir=input_dir, output_path=output_path)
    print(f"Wrote {n} chunks to {output_path}")