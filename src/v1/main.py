import os
import re
import jwt
import json
import logging
import shutil
import zipfile
from io import BytesIO
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from unidecode import unidecode
import markdown

from docx import Document
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request, WebSocket, status, WebSocketDisconnect
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session, configure_mappers
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_core.callbacks import BaseCallbackHandler
import pandas as pd
from models import Base, FileMetadata, User, Chat, Message
from auth import create_access_token, authenticate_user, get_current_user, get_password_hash, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta
from database import get_db
from jwt import ExpiredSignatureError

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
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

config = Config()

# Инициализация компонентов
engine = create_engine(config.POSTGRES_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SQLDatabase(engine=engine)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)
templates = Jinja2Templates(directory="templates")

# Создание необходимых директорий
os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

# Конфигурируем мапперы сразу после импорта моделей
logger.info("Configuring SQLAlchemy mappers")
configure_mappers()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекст жизненного цикла приложения"""
    global vectorstore
    logger.info("Creating database tables")
    Base.metadata.create_all(bind=engine)
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
app.mount("/static", StaticFiles(directory="static"), name="static")

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

def normalize_table_name(filename: str) -> str:
    if filename.startswith('._'):
        filename = filename[2:]
    try:
        filename = filename.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            filename = filename.encode('utf-8').decode('utf-8')
        except:
            pass
    name = os.path.splitext(filename)[0]
    name = unidecode(name)
    name = re.sub(r'[^\w]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    name = name.lower()
    if len(name) > config.MAX_TABLE_NAME_LENGTH:
        name = name[:config.MAX_TABLE_NAME_LENGTH]
    return name

def get_table_names(db: Session) -> List[str]:
    try:
        table_names = db.query(FileMetadata.table_name).filter(FileMetadata.table_name.isnot(None)).distinct().all()
        table_names = [t[0] for t in table_names]
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        return [t for t in table_names if t in existing_tables]
    except Exception as e:
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
            base_url="https://api.aitunnel.ru/v1/"
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
    with SessionLocal() as session:
        file_metadata = session.query(FileMetadata).all()
        file_info = "\n".join(
            f"- {row.file_name} ({row.file_type}) in {row.file_path or 'root'}"
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

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    print(request.url.path, 'request')
    # Разрешаем доступ к публичным страницам без аутентификации
    public_paths = ["/login", "/register", "/static", "/token"]
    if any(request.url.path.startswith(path) for path in public_paths):
        return await call_next(request)

    # Проверяем токен для защищенных страниц
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        request.state.user = payload.get("sub")
    except ExpiredSignatureError:
        return RedirectResponse(url="/login")

    response = await call_next(request)
    return response

@app.get("/")
async def root(request: Request):
    return RedirectResponse(url="/dashboard")

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/dashboard")
async def dashboard(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_user": current_user
    })

@app.get("/upload")
async def upload_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "current_user": current_user
    })

@app.get("/files")
async def files_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("files.html", {
        "request": request,
        "current_user": current_user
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = get_password_hash(password)
    user = User(username=username, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"username": user.username}

@app.post("/token")
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return JSONResponse(content={"access_token": access_token, "token_type": "bearer"})

async def process_xlsx(file: UploadFile, file_path: str = "", db: Session = Depends(get_db)):
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
        df = pd.read_excel(temp_path, engine="openpyxl")
        table_name = normalize_table_name(file.filename)
        try:
            file_name = file.filename.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                file_name = file.filename.encode('utf-8').decode('utf-8')
            except:
                pass
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        file_metadata = FileMetadata(
            file_name=file_name,
            file_path=file_path,
            file_type="xlsx",
            table_name=table_name,
            file_size=len(content)
        )
        db.add(file_metadata)
        db.commit()
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

async def process_docx(file: UploadFile, file_path: str = "", db: Session = Depends(get_db)):
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
        file_metadata = FileMetadata(
            file_name=file_name,
            file_path=file_path,
            file_type="docx",
            file_size=len(content)
        )
        db.add(file_metadata)
        db.commit()
        text_path = os.path.join(config.DOCUMENTS_DIR, f"{file_name}.txt")
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

@app.post("/upload_xlsx", dependencies=[Depends(get_current_user)])
async def upload_xlsx(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .xlsx files are supported"
        )
    return await process_xlsx(file, db=db)

@app.post("/upload_docx", dependencies=[Depends(get_current_user)])
async def upload_docx(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .docx files are supported"
        )
    return await process_docx(file, db=db)

@app.post("/upload_folder", dependencies=[Depends(get_current_user)])
async def upload_folder(zip_file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP archives are supported")
    temp_dir = f"temp_{os.urandom(4).hex()}"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        content = await zip_file.read()
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="ZIP file size exceeds maximum allowed size"
            )
        with zipfile.ZipFile(BytesIO(content)) as zip_ref:
            file_list = []
            for file_info in zip_ref.infolist():
                try:
                    corrected_filename = file_info.filename.encode('cp437').decode('utf-8')
                except:
                    corrected_filename = file_info.filename
                if not corrected_filename.startswith('__MACOSX/') and not corrected_filename.startswith(
                        '._') and not corrected_filename.endswith('/'):
                    file_list.append((corrected_filename, file_info))
            results = []
            for filename, file_info in file_list:
                try:
                    safe_filename = filename.replace('/', '_')
                    extract_path = os.path.join(temp_dir, safe_filename)
                    with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    rel_path = os.path.dirname(filename)
                    if filename.lower().endswith(".xlsx"):
                        with open(extract_path, "rb") as f:
                            result = await process_xlsx(
                                UploadFile(filename=os.path.basename(filename), file=BytesIO(f.read())),
                                rel_path,
                                db
                            )
                            results.append(result)
                    elif filename.lower().endswith(".docx"):
                        with open(extract_path, "rb") as f:
                            result = await process_docx(
                                UploadFile(filename=os.path.basename(filename), file=BytesIO(f.read())),
                                rel_path,
                                db
                            )
                            results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    results.append({
                        "status": "error",
                        "filename": filename,
                        "error": str(e)
                    })
        return {"message": "Folder processing completed", "results": results}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file format")
    except Exception as e:
        logger.error(f"Folder upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/file_metadata", dependencies=[Depends(get_current_user)])
async def get_file_metadata(path: str = None, file_type: str = None, db: Session = Depends(get_db)):
    query = db.query(FileMetadata)
    if path:
        query = query.filter(FileMetadata.file_path == path)
    if file_type:
        query = query.filter(FileMetadata.file_type == file_type.lower())
    return [row.__dict__ for row in query.all()]

@app.post("/search_docs", dependencies=[Depends(get_current_user)])
async def search_documents(query: str, path: str = None):
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

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        token = websocket.headers.get("authorization", "").replace("Bearer ", "")
        if not token:
            await websocket.send_json({
                "type": "error",
                "message": "JWT token is missing",
                "status": "error"
            })
            return
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid token",
                    "status": "error"
                })
                return
        except jwt.PyJWTError:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid JWT token",
                "status": "error"
            })
            return
        user = db.query(User).filter(User.username == username).first()
        if not user:
            await websocket.send_json({
                "type": "error",
                "message": "User not found",
                "status": "error"
            })
            return
        data = await websocket.receive_json()
        question = data["question"]
        model = data.get("model", "gemini")
        await websocket.send_json({"type": "progress", "message": "Обработка запроса..."})
        if cached := get_cached_response(question, model):
            # Парсим ответ из кэша в HTML
            html_content = markdown.markdown(cached["answer"], extensions=['extra', 'nl2br'])
            await websocket.send_json({
                "type": "answer",
                "answer": cached["answer"],
                "html_answer": html_content,
                "status": "complete"
            })
            return
        table_names = get_table_names(db)
        if not table_names:
            await websocket.send_json({
                "type": "answer",
                "answer": "No tables found. Please upload data first.",
                "html_answer": "<p>No tables found. Please upload data first.</p>",
                "status": "complete"
            })
            return
        llm = get_llm(model)
        agent = create_sql_agent_wrapper(llm, db, table_names)
        agent.callbacks = [WebSocketCallbackHandler(websocket)]
        answer = agent.run(question)
        # Парсим ответ в HTML
        html_content = markdown.markdown(answer, extensions=['extra', 'nl2br'])
        cache_response(question, model, {"answer": answer})
        await websocket.send_json({
            "type": "answer",
            "answer": answer,
            "html_answer": html_content,
            "status": "complete"
        })
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket query failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Ошибка: {str(e)}",
            "status": "error"
        })
    finally:
        await websocket.close()

@app.post("/chats", dependencies=[Depends(get_current_user)])
async def create_chat(title: str = Form(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    chat = Chat(title=title, user_id=current_user.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id, "title": chat.title}

@app.get("/chats", dependencies=[Depends(get_current_user)])
def list_chats(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title} for c in chats]

@app.get("/chats/{chat_id}/messages", dependencies=[Depends(get_current_user)])
async def get_chat_messages(chat_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return [
        {
            "sender": m.sender,
            "content": m.content,
            "html_content": markdown.markdown(m.content, extensions=['extra', 'nl2br']),
            "timestamp": m.timestamp.isoformat()
        } for m in chat.messages
    ]

@app.post("/chats/{chat_id}/messages", dependencies=[Depends(get_current_user)])
async def send_message(chat_id: int, message: str = Form(...), model: str = Form("chatgpt"),
                      db_session: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    chat = db_session.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    user_msg = Message(chat_id=chat.id, sender="user", content=message)
    db_session.add(user_msg)
    db_session.commit()

    # Подготовка истории
    messages = db_session.query(Message).filter(Message.chat_id == chat.id).order_by(Message.timestamp).all()
    context = [{"role": "system", "content": "Ты — полезный помощник, помогающий анализировать данные."}]
    for msg in messages:
        role = "user" if msg.sender == "user" else "assistant"
        context.append({"role": role, "content": msg.content})

    # Получаем список таблиц
    table_names = get_table_names(db_session)
    if not table_names:
        raise HTTPException(status_code=400, detail="No tables found. Please upload data first.")

    # Создаем SQL-агент с глобальным db (SQLDatabase)
    llm = get_llm(model)
    agent = create_sql_agent_wrapper(llm, db, table_names)

    # Формируем запрос с учетом контекста чата
    query = f"Вопрос: {message}\n\nИстория чата:\n" + "\n".join(
        f"{msg.sender}: {msg.content}" for msg in messages
    )

    # Выполняем запрос через SQL-агент
    try:
        response = agent.run(query)
        ai_content = str(response)
        # Парсим ответ в HTML
        html_content = markdown.markdown(ai_content, extensions=['extra', 'nl2br'])
    except Exception as e:
        logger.error(f"Error processing query with SQL agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    ai_msg = Message(chat_id=chat.id, sender="ai", content=ai_content)
    db_session.add(ai_msg)
    db_session.commit()

    return {"reply": ai_content, "reply_html": html_content}
