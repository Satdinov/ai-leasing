import logging

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import Config
from app.routes.auth import router as auth_router
from app.routes.root import router as root_router
from app.routes.files import router as files_router
from app.routes.chats import router as chats_router
from app.routes.inn_checker import router as inn_checker_router
from app.routes.document_generator import router as doc_generator_router
from app.routes.application import router as applications_router
from app.models import Base
from app.database import engine
from jwt import ExpiredSignatureError, decode as jwt_decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Leasing Company Tools")  # ✅ Сменили название

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/company_reports", StaticFiles(directory="company_reports"), name="reports")

Base.metadata.create_all(bind=engine)

app.include_router(auth_router, prefix="/auth")
app.include_router(inn_checker_router)
app.include_router(doc_generator_router)
app.include_router(applications_router)
app.include_router(files_router, prefix="/files")
app.include_router(chats_router, prefix="/chats")
app.include_router(root_router)
app.mount("/reports", StaticFiles(directory="company_reports"), name="reports")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    public_paths = ["/login", "/register", "/static", "/auth/token", "/auth/login", "/auth/register"]
    is_public = any(request.url.path.startswith(path) for path in public_paths)

    token = request.cookies.get("access_token")

    if not token and not is_public:
        return RedirectResponse(url="/auth/login")

    if token:
        try:
            payload = jwt_decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
            request.state.user = payload.get("sub")
        except ExpiredSignatureError:
            if not is_public:
                return RedirectResponse(url="/auth/login")

    response = await call_next(request)
    return response
