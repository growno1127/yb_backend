import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent
DATA = ROOT / "data"
MAX_SENTENCE_LENGTH = 100

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "youtube_videos"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Youtube
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_URL = os.getenv("YOUTUBE_API_URL")