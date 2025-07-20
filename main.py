import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from docx import Document
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Form, Request, WebSocket, status
from fastapi.security import APIKeyHeader
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_core.callbacks import BaseCallbackHandler
import pandas as pd

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
load_dotenv()


# Конфигурация приложения
class Config:
    POSTGRES_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
                   f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    PERSIST_DIRECTORY = "chroma_db"
    API_KEY = os.getenv("API_KEY")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    DOCUMENTS_DIR = "documents"
    CACHE_DIR = "cache"


config = Config()

# Инициализация компонентов
engine = create_engine(config.POSTGRES_URI)
db = SQLDatabase(engine=engine)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
templates = Jinja2Templates(directory="templates")

# Создание необходимых директорий
os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекст жизненного цикла приложения"""
    # Инициализация векторного хранилища при запуске
    global vectorstore
    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/"
        )
    )
    yield
    # Очистка ресурсов при остановке
    if vectorstore:
        vectorstore.delete_collection()
        vectorstore = None


app = FastAPI(
    title="AI Agent for Business Automation",
    lifespan=lifespan
)


class WebSocketCallbackHandler(BaseCallbackHandler):
    """Обработчик событий для WebSocket"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_agent_action(self, action, **kwargs):
        if action.tool == "sql_db_query":
            await self.websocket.send_json(
                {"type": "progress", "message": f"Выполняется SQL-запрос: {action.tool_input}"})
        elif action.tool == "sql_db_schema":
            await self.websocket.send_json(
                {"type": "progress", "message": f"Проверка схемы таблицы: {action.tool_input}"})
        elif action.tool == "sql_db_list_tables":
            await self.websocket.send_json({"type": "progress", "message": "Получение списка таблиц"})


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Проверка валидности API ключа"""
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key


def get_table_names() -> list[str]:
    """Получение списка таблиц в БД с проверкой регистра"""
    try:
        with engine.connect() as conn:
            # Получаем имена таблиц в нижнем регистре для сравнения
            result = conn.execute(text("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname = 'public'
            """))
            return [row[0] for row in result]
    except SQLAlchemyError as e:
        logger.error(f"Error getting table names: {e}")
        return []


def get_cache_path(model: str) -> str:
    """Генерация пути к файлу кэша"""
    return os.path.join(config.CACHE_DIR, f"query_cache_{model}.json")


def get_cached_response(question: str, model: str) -> Optional[Dict]:
    """Поиск ответа в кэше"""
    cache_path = get_cache_path(model)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
            return cache.get(question)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading cache: {e}")
        return None


def cache_response(question: str, model: str, response: Dict):
    """Сохранение ответа в кэш"""
    cache_path = get_cache_path(model)
    cache = {}

    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cache = json.load(f)

        cache[question] = response
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error writing cache: {e}")


def get_llm(model_name: str):
    """Фабрика LLM моделей"""
    models = {
        "gemini": lambda: ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        ),
        "chatgpt": lambda: ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/"
        ),
        "deepseek": lambda: ChatDeepSeek(
            model="deepseek-r1",
            api_key=os.getenv("AITUNNEL_API_KEY")
        )
    }

    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model_name}"
        )

    return models[model_name]()


def create_sql_agent_wrapper(llm, db, table_names: list[str]):
    """Создание SQL агента с проверкой существования таблиц"""
    try:
        # Проверяем существование таблиц в БД
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        # Фильтруем только существующие таблицы
        valid_tables = [t for t in table_names if t in existing_tables]

        if not valid_tables:
            raise HTTPException(
                status_code=400,
                detail=f"None of the tables {table_names} exist in database"
            )

        table_info = db.get_table_info(valid_tables)

        prefix = f"""
        You are querying a database with tables: {valid_tables}.
        Schema: {table_info}.
        Always exclude summary rows labeled 'Итого' by adding WHERE "<driver_column>" IS NOT NULL AND "<driver_column>" != 'Итого' to your SQL queries.
        """

        return create_sql_agent(
            llm=llm,
            db=db,
            agent_type="openai-tools",
            verbose=True,
            include_tables=valid_tables,
            prefix=prefix,
            max_iterations=10
        )
    except Exception as e:
        logger.error(f"Error creating SQL agent: {e}")
        raise

@app.get("/", dependencies=[Depends(verify_api_key)])
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": None
    })


@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query(
        request: Request,
        question: str = Form(...),
        model: str = Form("gemini")
):
    try:
        # Проверка кэша
        if cached := get_cached_response(question, model):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "answer": cached["answer"],
                "question": question,
                "model": model
            })

        # Получение таблиц и LLM
        table_names = get_table_names()
        if not table_names:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "answer": "No tables found. Please upload XLSX first.",
                "question": question,
                "model": model
            })

        llm = get_llm(model)
        agent = create_sql_agent_wrapper(llm, db, table_names)
        answer = agent.run(question)

        # Кэширование результата
        cache_response(question, model, {"answer": answer})

        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": answer,
            "question": question,
            "model": model
        })

    except Exception as e:
        logger.exception(f"Query failed: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": f"Error: {str(e)}",
            "question": question,
            "model": model
        })


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        question = data["question"]
        print('data', data)
        print('question', question)
        model = data.get("model", "chatgpt")

        await websocket.send_json({"type": "progress", "message": "Обработка запроса..."})

        if cached := get_cached_response(question, model):
            await websocket.send_json({
                "type": "answer",
                "answer": cached["answer"],
                "status": "complete"
            })
            await websocket.close()
            return

        table_names = get_table_names()
        if not table_names:
            await websocket.send_json({
                "type": "answer",
                "answer": "No tables found. Please upload XLSX first.",
                "status": "complete"
            })
            await websocket.close()
            return

        # Добавляем лог для отладки
        logger.info(f"Available tables: {table_names}")

        llm = get_llm(model)
        try:
            agent = create_sql_agent_wrapper(llm, db, table_names)
            answer = agent.run(question)

            cache_response(question, model, {"answer": answer})

            await websocket.send_json({
                "type": "answer",
                "answer": answer,
                "status": "complete"
            })
        except HTTPException as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e.detail),
                "status": "error"
            })

    except Exception as e:
        logger.exception(f"WebSocket query failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Ошибка: {str(e)}",
            "status": "error"
        })
    finally:
        await websocket.close()

@app.post("/upload_xlsx", dependencies=[Depends(verify_api_key)])
async def upload_xlsx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .xlsx allowed"
        )

    temp_path = f"temp_{file.filename}"
    try:
        # Сохранение временного файла
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Чтение и обработка Excel
        df = pd.read_excel(temp_path)
        table_name = os.path.splitext(file.filename)[0].replace(" ", "_")

        # Загрузка в PostgreSQL
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False
        )

        return {"message": f"Report {file.filename} uploaded to PostgreSQL"}

    except Exception as e:
        logger.error(f"XLSX upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/upload_docx", dependencies=[Depends(verify_api_key)])
async def upload_docx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .docx allowed"
        )

    temp_path = f"temp_{file.filename}"
    try:
        # Сохранение временного файла
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Извлечение текста из DOCX
        doc = Document(temp_path)
        text_content = "\n".join(para.text for para in doc.paragraphs if para.text)

        # Векторизация текста
        split_texts = text_splitter.split_text(text_content)
        vectorstore.add_texts(
            texts=split_texts,
            collection_name="docx_data"
        )

        # Сохранение текстовой копии
        text_path = os.path.join(
            config.DOCUMENTS_DIR,
            f"{file.filename}.txt"
        )
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        return {"message": f"Document {file.filename} processed and vectorized"}

    except Exception as e:
        logger.error(f"DOCX upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)