import json
import logging
import os
import re
from typing import Dict, List, Optional

from fastapi import HTTPException, status
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from unidecode import unidecode

from app.config import Config
from app.database import engine
from app.models import FileMetadata

logger = logging.getLogger(__name__)


def normalize_table_name(filename: str) -> str:
    if filename.startswith("._"):
        filename = filename[2:]
    try:
        filename = filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        filename = filename.encode("utf-8").decode("utf-8")
    name = os.path.splitext(filename)[0]
    name = unidecode(name)
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    name = name.lower()
    if len(name) > Config.MAX_TABLE_NAME_LENGTH:
        name = name[: Config.MAX_TABLE_NAME_LENGTH]
    return name


def get_table_names(db: Session) -> List[str]:
    try:
        table_names = (
            db.query(FileMetadata.table_name)
            .filter(FileMetadata.table_name.isnot(None))
            .distinct()
            .all()
        )
        table_names = [t[0] for t in table_names]
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        return [t for t in table_names if t in existing_tables]
    except Exception as e:
        logger.error(f"Error getting table names: {e}")
        return []


def get_cache_path(model: str) -> str:
    return os.path.join(Config.CACHE_DIR, f"query_cache_{model}.json")


def get_cached_response(question: str, model: str) -> Optional[Dict]:
    cache_path = get_cache_path(model)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f).get(question)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading cache: {e}")
        return None


def cache_response(question: str, model: str, response: Dict):
    cache_path = get_cache_path(model)
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading cache: {e}")
    cache[question] = response
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error writing cache: {e}")


def get_llm(model_name: str):
    models = {
        "gemini": lambda: ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/",
        ),
        "chatgpt": lambda: ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/",
            temperature=0,
        ),
        "deepseek": lambda: ChatDeepSeek(
            model="deepseek-r1",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/",
        ),
    }
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model_name}",
        )
    return models[model_name]()
