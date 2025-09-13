import os
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.auth import get_current_user
from app.models import User
from app.services.document_generator_service import generate_application

router = APIRouter(prefix="/api/documents", tags=["document_generator"])


class Supplier(BaseModel):
    company: Optional[str] = None
    inn: Optional[str] = None
    address: Optional[str] = None


class Asset(BaseModel):
    name: Optional[str] = None
    quantity: Optional[int] = None
    delivery_time: Optional[int] = None
    total_cost: Optional[str] = None


class Guarantor(BaseModel):
    full_name: Optional[str] = None
    inn: Optional[str] = None
    contacts: Optional[str] = None


class Pledge(BaseModel):
    name: Optional[str] = None
    quantity: Optional[int] = None
    market_value: Optional[str] = None


class DealData(BaseModel):
    lessee_company: Optional[str] = None
    lessee_inn: Optional[str] = None
    lessee_legal_address: Optional[str] = None
    lessee_actual_address: Optional[str] = None
    lessee_director: Optional[str] = None
    asset_term: Optional[int] = None
    advance_payment_percent: Optional[int] = None
    suppliers: List[Supplier] = []
    assets: List[Asset] = []
    guarantors: List[Guarantor] = []
    pledges: List[Pledge] = []


@router.post("/generate/application")
async def generate_application_route(
    deal_data: DealData, current_user: User = Depends(get_current_user)
):
    deal_data_dict = deal_data.model_dump()
    print(deal_data_dict)
    file_path = generate_application(deal_data_dict)

    if not file_path:
        raise HTTPException(
            status_code=500, detail="Не удалось сгенерировать документ."
        )

    filename = os.path.basename(file_path)
    encoded_filename = quote(filename)
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
    }

    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )
