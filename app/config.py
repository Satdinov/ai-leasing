import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    POSTGRES_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
                   f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    PERSIST_DIRECTORY = "chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    DOCUMENTS_DIR = "documents"
    CACHE_DIR = "cache"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_TABLE_NAME_LENGTH = 63  # Ограничение PostgreSQL
