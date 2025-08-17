# app/routes/inn_checker.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.services.leasing_service import get_company_info_by_inn # ✅ Обновленный импорт
from app.auth import get_current_user
from app.models import User

# ✅ Префикс изменен на /api/inn для ясности, что это API-эндпоинт
router = APIRouter(prefix="/api/inn", tags=["inn_checker"])

class InnCheckRequest(BaseModel):
    inn: str = Field(..., description="ИНН компании для проверки")

@router.post("/check") # ✅ Эндпоинт тоже стал короче
async def check_inn_route(request: InnCheckRequest, current_user: User = Depends(get_current_user)):
    result = await get_company_info_by_inn(request.inn)
    if not result.get("is_success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Произошла ошибка"))
    return result.get("data")