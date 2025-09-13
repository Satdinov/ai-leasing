import os
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.auth import get_current_user
from app.models import User
from app.schemas.schemas import DealData
from app.services.document_generator_service import generate_application

router = APIRouter(prefix="/api/documents", tags=["document_generator"])


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
