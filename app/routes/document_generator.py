# app/routes/document_generator.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
import os
from urllib.parse import quote

from app.services.document_generator_service import generate_application
from app.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/api/documents", tags=["document_generator"])


# ✅ НОВЫЕ И ОБНОВЛЕННЫЕ МОДЕЛИ
class Supplier(BaseModel):
    company: str
    inn: str
    address: str


class Asset(BaseModel):
    name: str
    quantity: int
    delivery_time: int
    total_cost: str  # Используем строку, чтобы можно было передавать форматированные числа


class Guarantor(BaseModel):
    full_name: str
    inn: str
    contacts: str

class Pledge(BaseModel):
    name: str
    quantity: int
    market_value: str

class DealData(BaseModel):
    lessee_company: str
    lessee_inn: str
    lessee_legal_address: str
    lessee_actual_address: str
    lessee_director: str
    asset_term: int
    advance_payment_percent: int
    suppliers: List[Supplier]
    assets: List[Asset]
    guarantors: List[Guarantor]
    pledges: List[Pledge]

@router.post("/generate/application")
async def generate_application_route(deal_data: DealData, current_user: User = Depends(get_current_user)):
    deal_data_dict = deal_data.model_dump()
    print(deal_data_dict)
    file_path = generate_application(deal_data_dict)

    if not file_path:
        raise HTTPException(status_code=500, detail="Не удалось сгенерировать документ.")

    filename = os.path.basename(file_path)
    encoded_filename = quote(filename)
    headers = {'Content-Disposition': f'attachment; filename*=UTF-8\'\'{encoded_filename}'}

    return FileResponse(
        path=file_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        headers=headers
    )
