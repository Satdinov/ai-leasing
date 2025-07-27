from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from models import User
from auth import get_current_user

router = APIRouter(tags=["root"])
templates = Jinja2Templates(directory="templates")

@router.get("/")
async def root(request: Request):
    return RedirectResponse(url="/dashboard")

@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.get("/dashboard")
async def dashboard(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_user": current_user
    })

@router.get("/upload")
async def upload_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "current_user": current_user
    })

@router.get("/files")
async def files_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("files.html", {
        "request": request,
        "current_user": current_user
    })
