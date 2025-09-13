from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

import app.models as models
from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.routes.document_generator import DealData

router = APIRouter(prefix="/api/applications", tags=["applications"])


class ApplicationInfo(BaseModel):
    id: int
    title: str
    updated_at: str

    class Config:
        from_attributes = True


@router.post("/save")
async def save_application(
    application_data: DealData,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    new_application = models.LeasingApplication(
        title=f"Заявка от {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        user_id=current_user.id,
        lessee_company=application_data.lessee_company,
        lessee_inn=application_data.lessee_inn,
        lessee_legal_address=application_data.lessee_legal_address,
        lessee_actual_address=application_data.lessee_actual_address,
        lessee_director=application_data.lessee_director,
        asset_term=application_data.asset_term,
        advance_payment_percent=application_data.advance_payment_percent,
    )

    new_application.suppliers = [
        models.Supplier(**s.model_dump()) for s in application_data.suppliers
    ]
    new_application.assets = [
        models.Asset(**a.model_dump()) for a in application_data.assets
    ]
    new_application.guarantors = [
        models.Guarantor(**g.model_dump()) for g in application_data.guarantors
    ]
    new_application.pledges = [
        models.Pledge(**p.model_dump()) for p in application_data.pledges
    ]

    db.add(new_application)
    db.commit()
    db.refresh(new_application)

    return {"status": "success", "application_id": new_application.id}


@router.put("/{application_id}")
async def update_application(
    application_id: int,
    application_data: DealData,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    app = (
        db.query(models.LeasingApplication)
        .filter(
            models.LeasingApplication.id == application_id,
            models.LeasingApplication.user_id == current_user.id,
        )
        .first()
    )

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Заявка не найдена"
        )

    app.lessee_company = application_data.lessee_company
    app.lessee_inn = application_data.lessee_inn
    app.lessee_legal_address = application_data.lessee_legal_address
    app.lessee_actual_address = application_data.lessee_actual_address
    app.lessee_director = application_data.lessee_director
    app.asset_term = application_data.asset_term
    app.advance_payment_percent = application_data.advance_payment_percent
    app.updated_at = datetime.utcnow()

    app.suppliers = [
        models.Supplier(**s.model_dump()) for s in application_data.suppliers
    ]
    app.assets = [models.Asset(**a.model_dump()) for a in application_data.assets]
    app.guarantors = [
        models.Guarantor(**g.model_dump()) for g in application_data.guarantors
    ]
    app.pledges = [models.Pledge(**p.model_dump()) for p in application_data.pledges]

    db.commit()
    db.refresh(app)
    return {"status": "success", "application_id": app.id}


@router.get("/", response_model=List[ApplicationInfo])
async def get_applications(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    applications = (
        db.query(models.LeasingApplication)
        .filter(models.LeasingApplication.user_id == current_user.id)
        .order_by(models.LeasingApplication.updated_at.desc())
        .all()
    )
    return [
        {
            "id": app.id,
            "title": app.title,
            "updated_at": app.updated_at.strftime("%Y-%m-%d %H:%M"),
        }
        for app in applications
    ]


@router.get("/{application_id}", response_model=DealData)
async def get_application_details(
    application_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    app = (
        db.query(models.LeasingApplication)
        .filter(
            models.LeasingApplication.id == application_id,
            models.LeasingApplication.user_id == current_user.id,
        )
        .first()
    )

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Заявка не найдена"
        )

    return app


@router.delete("/{application_id}")
async def delete_application(
    application_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    app = (
        db.query(models.LeasingApplication)
        .filter(
            models.LeasingApplication.id == application_id,
            models.LeasingApplication.user_id == current_user.id,
        )
        .first()
    )

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Заявка не найдена"
        )

    db.delete(app)
    db.commit()
    return {"status": "success", "message": "Заявка удалена"}
