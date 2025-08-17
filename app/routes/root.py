# app/routes/root.py
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from app.models import User
from app.auth import get_current_user

router = APIRouter(tags=["root"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/")
async def root(request: Request):
    # ✅ Главная страница теперь перенаправляет на /dashboard
    return RedirectResponse(url="/dashboard")

@router.get("/dashboard")
async def dashboard_page(request: Request, current_user: User = Depends(get_current_user)):
    # ✅ Это теперь наша основная страница после входа
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_user": current_user
    })

# app/routes/root.py
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from app.models import User
from app.auth import get_current_user

router = APIRouter(tags=["root"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/")
async def root():
    return RedirectResponse(url="/dashboard")

@router.get("/dashboard")
async def dashboard_page(request: Request, current_user: User = Depends(get_current_user)):
    # Эта страница теперь используется для формы заявок
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_user": current_user
    })

@router.get("/chat")
async def chat_page(request: Request, current_user: User = Depends(get_current_user)):
    # Новый маршрут для страницы чата
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "current_user": current_user
    })

@router.get("/files")
async def files_page(request: Request, current_user: User = Depends(get_current_user)):
    # Маршрут для страницы с файлами
    return templates.TemplateResponse("files.html", {
        "request": request,
        "current_user": current_user
    })

@router.get("/upload")
async def upload_page(request: Request, current_user: User = Depends(get_current_user)):
    # Маршрут для страницы загрузки
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "current_user": current_user
    })

@router.get("/check-inn")
async def check_inn_page(request: Request, current_user: User = Depends(get_current_user)):
    # ✅ Новый маршрут для страницы проверки ИНН
    return templates.TemplateResponse("check_inn.html", {
        "request": request,
        "current_user": current_user
    })