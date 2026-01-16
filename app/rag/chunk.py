import re
from typing import List, Tuple

HEADER_RE = re.compile(r"^#{1,6}\s+(.*)\s*$")


def split_sections(md: str) -> List[Tuple[List[str], str]]:
    """
    Split markdown into (section_path, section_text).
    Uses markdown headings (# .. ######) to build hierarchy.
    """
    lines = md.splitlines()
    sections = []
    current_buf: List[str] = []
    stack: List[Tuple[int, str]] = []
    current_path: List[str] = ["(root)"]

    def flush():
        nonlocal current_buf
        text = "\n".join(current_buf).strip()
        if text:
            sections.append((current_path.copy(), text))
        current_buf = []

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            title = m.group(1).strip()
            level = len(line) - len(line.lstrip("#"))

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = [t for _, t in stack]
        else:
            current_buf.append(line)

    flush()
    return sections


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if max_chars <= overlap:
        raise ValueError("max_chars must be > overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks