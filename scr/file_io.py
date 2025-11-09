# -----------------------
# IO
# -----------------------
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_paper(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        out.append(text[i : i + size])
        i += step
    return out

def approx_char_start(chunk_index: int) -> int:
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    return max(0, chunk_index * step)

