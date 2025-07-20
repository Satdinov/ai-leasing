import os
import re
import json
import logging
import shutil
import zipfile
from io import BytesIO
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from unidecode import unidecode

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

class Config:
    POSTGRES_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
                   f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    PERSIST_DIRECTORY = "chroma_db"
    API_KEY = os.getenv("API_KEY")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    DOCUMENTS_DIR = "documents"
    CACHE_DIR = "cache"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_TABLE_NAME_LENGTH = 63  # Ограничение PostgreSQL

config = Config()

# Инициализация компонентов
engine = create_engine(config.POSTGRES_URI)

def init_db():
    """Инициализация таблиц БД"""
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS file_metadata (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            table_name TEXT,
            file_size INTEGER,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        conn.commit()

init_db()

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
    global vectorstore
    try:
        vectorstore = Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("AITUNNEL_API_KEY"),
                base_url="https://api.aitunnel.ru/v1/"
            )
        )
        yield
    finally:
        if vectorstore:
            try:
                vectorstore.delete_collection()
            except Exception as e:
                logger.error(f"Error cleaning up vectorstore: {e}")

app = FastAPI(
    title="AI Agent for Business Automation",
    lifespan=lifespan
)

class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_agent_action(self, action, **kwargs):
        if action.tool == "sql_db_query":
            await self.websocket.send_json(
                {"type": "progress", "message": f"Выполняется SQL-запрос: {action.tool_input}"})
        elif action.tool == "sql_db_schema":
            await self.websocket.send_json(
                {"type": "progress", "message": f"Проверка схемы таблицы: {action.tool_input}"})

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

def normalize_table_name(filename: str) -> str:
    """
    Нормализует имя таблицы:
    1. Исправляет кодировку имен файлов из ZIP
    2. Удаляет расширение файла
    3. Транслитерирует кириллицу в латиницу
    4. Заменяет спецсимволы на подчеркивания
    5. Усекает до максимальной длины
    """
    # Исправление кодировки для имен из ZIP
    if filename.startswith('._'):
        filename = filename[2:]

    try:
        filename = filename.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            filename = filename.encode('utf-8').decode('utf-8')
        except:
            pass

    # Удаляем расширение файла
    name = os.path.splitext(filename)[0]

    # Транслитерируем кириллицу в латиницу
    name = unidecode(name)

    # Заменяем все не-буквенно-цифровые символы на подчеркивания
    name = re.sub(r'[^\w]', '_', name)

    # Удаляем повторяющиеся подчеркивания
    name = re.sub(r'_+', '_', name)

    # Удаляем подчеркивания в начале и конце
    name = name.strip('_')

    # Приводим к нижнему регистру
    name = name.lower()

    # Усекаем до максимальной длины
    if len(name) > config.MAX_TABLE_NAME_LENGTH:
        name = name[:config.MAX_TABLE_NAME_LENGTH]

    return name

def get_table_names() -> List[str]:
    """Получение списка таблиц с проверкой существования"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT table_name 
                FROM file_metadata 
                WHERE table_name IS NOT NULL
            """))
            table_names = [row[0] for row in result if row[0]]

            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            return [t for t in table_names if t in existing_tables]
    except SQLAlchemyError as e:
        logger.error(f"Error getting table names: {e}")
        return []

def get_cache_path(model: str) -> str:
    return os.path.join(config.CACHE_DIR, f"query_cache_{model}.json")

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
            base_url = "https://api.aitunnel.ru/v1/"
        ),
        "chatgpt": lambda: ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/"
        ),
        "deepseek": lambda: ChatDeepSeek(
            model="deepseek-r1",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/"
        )
    }
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model_name}"
        )
    return models[model_name]()

def create_sql_agent_wrapper(llm, db, table_names: List[str]):
    """Создание SQL агента с информацией о файлах"""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT file_name, file_path, file_type, table_name 
            FROM file_metadata
        """))
        file_metadata = [
            {
                "file_name": row[0],
                "file_path": row[1],
                "file_type": row[2],
                "table_name": row[3]
            } for row in result
        ]

    file_info = "\n".join(
        f"- {row['file_name']} ({row['file_type']}) in {row['file_path'] or 'root'}"
        for row in file_metadata
    )

    table_info = db.get_table_info(table_names)

    prefix = f"""
    You are querying a database with tables: {table_names}.
    Schema: {table_info}.

    Available files and their locations:
    {file_info}

    Context:
    - **Revenue**: Consists of cash and non-cash trips.
    - **Expenses**: Include diesel fuel, truck/trailer repairs, driver salaries, trip allowances (used for repairs, fines, parking, etc.), OSAGO/KASKO insurance, cargo liability insurance, Platon (toll for federal roads), and toll roads (e.g., Moscow-Kazan highway).
    - **Mileage**: Monthly vehicle mileage is critical. Drivers may switch vehicles if one is under repair.
    - **Key Metrics**:
      1. **Revenue per kilometer**: Revenue/mileage (rubles/km). Higher is better.
      2. **Fuel consumption**: (Liters of diesel per month / mileage per month) × 100 (liters/100km). Higher means more expenses.
      3. **Repair cost per kilometer**: Repair costs per month / mileage per month (rubles/km).
      4. **Driver salary**: Calculated as mileage × 15 rubles.

    Key files and folders:
    - **Files**:
      - 'Список техники перевозчика и водителей ООО РАД-ТРАК.xlsx': Contains vehicle and driver details.
      - 'Годовой отчет 01.01-31.12.24.xlsx': Annual financial report for 2024.
      - 'Водители пробеги в 2024.xlsx': Driver mileage data for 2024.
    - **Folders**:
      - '1. Движение парка': Vehicle movement data.
      - '2. Заправки': Fueling data.
      - '3. Ремонт по месяцам': Monthly repair data (e.g., '01_remont_ianvar_24' for January, '08_remont_avgust_24' for August, '09_remont_sentiabr_24' for September).
      - '4. Ремонт по ТС': Repair data by vehicle.

    Answer these queries when asked:
    1. **Most effective driver**: Rank drivers by revenue generated and by mileage (who drives the most) using 'Водители пробеги в 2024' and 'Годовой отчет 01.01-31.12.24'.
    2. **Repair costs in a specific month**: For January, use only '01_remont_ianvar_24'; for August, use only '08_remont_avgust_24'; for September, use only '09_remont_sentiabr_24'. Identify vehicles with highest/lowest repair costs, excluding rows where 'Автомобиль' or 'Гос. № тягача / прицепа' is 'Итого', NULL, or empty (''). Ensure the repair cost column ('Итого', 'Сумма работ', or 'Сумма документа') is not NULL or empty.
    3. **Fuel consumption**: Identify trucks with highest/lowest fuel consumption (liters/100km) using '2. Заправки' and 'Водители пробеги в 2024'.
    4. **Repair cost per kilometer**: Calculate for Mercedes, Scania, and LOHR trailers using '4. Ремонт по ТС' and 'Водители пробеги в 2024' (repair cost/mileage).

    Important rules:
    1. **Always exclude summary rows** where 'Автомобиль' or 'Гос. № тягача / прицепа' is 'Итого', NULL, or empty ('') in all repair-related queries.
    2. **Exclude rows with NULL or empty repair costs** in columns like 'Итого', 'Сумма работ', or 'Сумма документа' for repair queries.
    3. For month-specific queries (e.g., January repairs), use only the relevant table (e.g., '01_remont_ianvar_24').
    4. When asked about file locations, check the file metadata.
    5. For driver-related queries, use columns from 'Список техники перевозчика и водителей ООО РАД-ТРАК' and 'Водители пробеги в 2024'.
    6. For repairs, use '3. Ремонт по месяцам' for monthly data and '4. Ремонт по ТС' for vehicle-specific data.
    7. For fuel, use '2. Заправки'.
    8. For revenue/expenses, use 'Годовой отчет 01.01-31.12.24'.
    9. When showing file paths, provide full path information.
    10. For vehicle-specific queries (e.g., Mercedes, Scania), use columns like 'Автомобиль' or 'Гос. № тягача / прицепа' to identify the vehicle model.
    """

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        include_tables=table_names,
        prefix=prefix,
        max_iterations=15,
        handle_parsing_errors=True
    )

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": None
    })

async def process_xlsx(file: UploadFile, file_path: str = "") -> Dict:
    """Обработка XLSX файла с сохранением метаданных"""
    temp_path = f"temp_{file.filename}"
    file_name = file.filename
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            if len(content) > config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds maximum allowed size"
                )
            f.write(content)

        # Чтение Excel с указанием движка
        df = pd.read_excel(temp_path, engine="openpyxl")

        # Генерация нормализованного имени таблицы
        table_name = normalize_table_name(file.filename)

        try:
            file_name = file.filename.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                file_name = file.filename.encode('utf-8').decode('utf-8')
            except:
                pass

        # Загрузка в PostgreSQL
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False
        )

        with engine.connect() as conn:
            conn.execute(
                text("""
                INSERT INTO file_metadata 
                (file_name, file_path, file_type, table_name, file_size)
                VALUES (:name, :path, :type, :table, :size)
                """),
                {
                    "name": file_name,
                    "path": file_path,
                    "type": "xlsx",
                    "table": table_name,
                    "size": len(content)
                }
            )
            conn.commit()

        return {
            "status": "success",
            "filename": file_name,
            "table_name": table_name,
            "path": file_path
        }
    except Exception as e:
        logger.error(f"Error processing XLSX {file_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing {file_name}: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def process_docx(file: UploadFile, file_path: str = "") -> Dict:
    """Обработка DOCX файла с сохранением метаданных"""
    temp_path = f"temp_{file.filename}"
    file_name = file.filename
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            if len(content) > config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds maximum allowed size"
                )
            f.write(content)

        doc = Document(temp_path)
        text_content = "\n".join(para.text for para in doc.paragraphs if para.text)

        try:
            file_name = file.filename.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                file_name = file.filename.encode('utf-8').decode('utf-8')
            except:
                pass

        split_texts = text_splitter.split_text(text_content)
        vectorstore.add_texts(
            texts=split_texts,
            metadatas=[{
                "source": file_path,
                "filename": file_name,
                "file_type": "docx"
            } for _ in split_texts],
            collection_name="docx_data"
        )

        with engine.connect() as conn:
            conn.execute(
                text("""
                INSERT INTO file_metadata 
                (file_name, file_path, file_type, file_size)
                VALUES (:name, :path, :type, :size)
                """),
                {
                    "name": file_name,
                    "path": file_path,
                    "type": "docx",
                    "size": len(content)
                }
            )
            conn.commit()

        text_path = os.path.join(
            config.DOCUMENTS_DIR,
            f"{file_name}.txt"
        )
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        return {
            "status": "success",
            "filename": file_name,
            "path": file_path,
            "chunks": len(split_texts)
        }
    except Exception as e:
        logger.error(f"Error processing DOCX {file_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing {file_name}: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/upload_xlsx", dependencies=[Depends(verify_api_key)])
async def upload_xlsx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .xlsx files are supported"
        )
    return await process_xlsx(file)

@app.post("/upload_docx", dependencies=[Depends(verify_api_key)])
async def upload_docx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .docx files are supported"
        )
    return await process_docx(file)

@app.post("/upload_folder", dependencies=[Depends(verify_api_key)])
async def upload_folder(zip_file: UploadFile = File(...)):
    """Загрузка и обработка ZIP-архива с файлами"""
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP archives are supported"
        )

    temp_dir = f"temp_{os.urandom(4).hex()}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Читаем содержимое ZIP-файла в память
        content = await zip_file.read()
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="ZIP file size exceeds maximum allowed size"
            )

        # Используем BytesIO для работы с ZIP в памяти
        with zipfile.ZipFile(BytesIO(content)) as zip_ref:
            # Получаем список файлов, исправляя кодировку имен
            file_list = []
            for file_info in zip_ref.infolist():
                try:
                    # Пытаемся исправить кодировку имени файла
                    corrected_filename = file_info.filename.encode('cp437').decode('utf-8')
                except:
                    corrected_filename = file_info.filename

                # Пропускаем служебные файлы MacOS и директории
                if not corrected_filename.startswith('__MACOSX/') and not corrected_filename.startswith('._') and not corrected_filename.endswith('/'):
                    file_list.append((corrected_filename, file_info))

            # Обрабатываем файлы
            results = []
            for filename, file_info in file_list:
                try:
                    # Создаем корректные пути для извлечения
                    safe_filename = filename.replace('/', '_')
                    extract_path = os.path.join(temp_dir, safe_filename)

                    # Извлекаем файл
                    with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                    # Определяем относительный путь
                    rel_path = os.path.dirname(filename)

                    # Обрабатываем файл
                    if filename.lower().endswith(".xlsx"):
                        with open(extract_path, "rb") as f:
                            result = await process_xlsx(
                                UploadFile(filename=os.path.basename(filename), file=BytesIO(f.read())),
                                rel_path
                            )
                            results.append(result)
                    elif filename.lower().endswith(".docx"):
                        with open(extract_path, "rb") as f:
                            result = await process_docx(
                                UploadFile(filename=os.path.basename(filename), file=BytesIO(f.read())),
                                rel_path
                            )
                            results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    results.append({
                        "status": "error",
                        "filename": filename,
                        "error": str(e)
                    })

        return {
            "message": "Folder processing completed",
            "results": results
        }
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400,
            detail="Invalid ZIP file format"
        )
    except Exception as e:
        logger.error(f"Folder upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/file_metadata", dependencies=[Depends(verify_api_key)])
async def get_file_metadata(path: str = None, file_type: str = None):
    """Получение метаданных файлов"""
    query = "SELECT * FROM file_metadata WHERE 1=1"
    params = {}
    if path:
        query += " AND file_path = :path"
        params["path"] = path
    if file_type:
        query += " AND file_type = :file_type"
        params["file_type"] = file_type.lower()

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row) for row in result]

@app.post("/search_docs", dependencies=[Depends(verify_api_key)])
async def search_documents(query: str, path: str = None):
    """Поиск по документам с фильтром по пути"""
    filters = {"file_type": "docx"}
    if path:
        filters["source"] = path

    results = vectorstore.similarity_search(
        query,
        k=5,
        filter=filters,
        collection_name="docx_data"
    )

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "filename": doc.metadata.get("filename", "")
        }
        for doc in results
    ]

@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query(
        request: Request,
        question: str = Form(...),
        model: str = Form("gemini")
):
    try:
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
                "answer": "No tables found. Please upload data first.",
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
        model = data.get("model", "gemini")

        await websocket.send_json({"type": "progress", "message": "Обработка запроса..."})

        if cached := get_cached_response(question, model):
            await websocket.send_json({
                "type": "answer",
                "answer": cached["answer"],
                "status": "complete"
            })
            return

        table_names = get_table_names()
        if not table_names:
            await websocket.send_json({
                "type": "answer",
                "answer": "No tables found. Please upload data first.",
                "status": "complete"
            })
            return

        llm = get_llm(model)
        agent = create_sql_agent_wrapper(llm, db, table_names)
        agent.callbacks = [WebSocketCallbackHandler(websocket)]

        answer = agent.run(question)
        cache_response(question, model, {"answer": answer})

        await websocket.send_json({
            "type": "answer",
            "answer": answer,
            "status": "complete"
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