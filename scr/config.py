# -----------------------
# Configuration constants
# -----------------------
import os

GEN_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash-lite")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
TOP_K_SNIPPETS = int(os.environ.get("TOP_K", 4))
MAX_TOTAL_CHUNKS = int(os.environ.get("MAX_TOTAL_CHUNKS", "3000"))
MAX_CHARS_PER_FILE = int(os.environ.get("MAX_CHARS_PER_FILE", "300000"))

