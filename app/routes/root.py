from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.auth import get_current_user
from app.models import User

router = APIRouter(tags=["root"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def root():
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard")
async def dashboard_page(
    request: Request, current_user: User = Depends(get_current_user)
):
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "current_user": current_user}
    )


@router.get("/chat")
async def chat_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "chat.html", {"request": request, "current_user": current_user}
    )


@router.get("/files")
async def files_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "files.html", {"request": request, "current_user": current_user}
    )


@router.get("/upload")
async def upload_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "upload.html", {"request": request, "current_user": current_user}
    )


@router.get("/check-inn")
async def check_inn_page(
    request: Request, current_user: User = Depends(get_current_user)
):
    return templates.TemplateResponse(
        "check_inn.html", {"request": request, "current_user": current_user}
    )
