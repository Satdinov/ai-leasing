import logging
import os

from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from routes.auth import router as auth_router
from routes.files import router as files_router
from routes.chats import router as chats_router
from routes.root import router as root_router
from models import Base
from database import engine
from config import Config
from auth import SECRET_KEY, ALGORITHM
from jwt import ExpiredSignatureError, decode as jwt_decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent for Business Automation"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Создание необходимых директорий
os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# Создание таблиц в базе данных
logger.info("Configuring SQLAlchemy mappers")
Base.metadata.create_all(bind=engine)

# Подключение маршрутов
app.include_router(auth_router, prefix="/auth")
app.include_router(files_router, prefix="/files")
app.include_router(chats_router, prefix="/chats")
app.include_router(root_router)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    public_paths = ["/login", "/register", "/static", "/auth/token", "/auth/login", "/auth/register"]
    if any(request.url.path.startswith(path) for path in public_paths):
        return await call_next(request)

    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")

    try:
        payload = jwt_decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        request.state.user = payload.get("sub")
    except ExpiredSignatureError:
        return RedirectResponse(url="/login")

    return await call_next(request)
