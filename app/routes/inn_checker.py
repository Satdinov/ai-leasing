import os

import markdown
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.schemas import InnRequest
from app.services.delta_services import (
    find_delta_company_id,
    get_processed_delta_report,
)
from app.services.leasing_service import get_company_info_by_inn
from app.utils import get_llm

router = APIRouter(prefix="/api/inn", tags=["inn_checker"])


class InnCheckRequest(BaseModel):
    inn: str = Field(..., description="ИНН компании для проверки")


@router.post("/check")
async def check_inn_route(
    request: InnCheckRequest, current_user: User = Depends(get_current_user)
):
    result = await get_company_info_by_inn(request.inn)
    if not result.get("is_success"):
        raise HTTPException(
            status_code=404, detail=result.get("error", "Произошла ошибка")
        )
    return result.get("data")


@router.post("/find-delta-id")
async def find_delta_id_route(
    request: InnCheckRequest, current_user: User = Depends(get_current_user)
):
    """
    Находит ID компании в системе Дельта по ИНН.
    """
    company_id = await find_delta_company_id(request.inn)
    return {"inn": request.inn, "delta_company_id": company_id}


@router.post("/get-delta-report")
async def get_delta_report_route(
    request: InnRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Скачивает, очищает, векторизует отчет и возвращает очищенный файл.
    """
    file_path = await get_processed_delta_report(request.inn, db)

    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.post("/check-delta-ai")
async def check_delta_with_ai_route(
    request: InnCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Получает отчет из Дельты, анализирует его с помощью ИИ
    и возвращает готовое заключение.
    """
    try:
        file_path, report_text = await get_processed_delta_report(request.inn, db)

        ai_prompt = f"""
        **Роль:** Ты — ведущий риск-аналитик в лизинговой компании.
        **Задача:** На основе представленного отчета по компании (ИНН {request.inn}) подготовь краткое, но емкое аналитическое заключение для внутреннего использования.

        **Структура заключения:**
        1.  **Общий вывод:** Начни с общего вывода о благонадежности компании (высокая, средняя, низкая, критическая).
        2.  **Ключевые позитивные факторы:** Выдели 2-3 основных положительных момента (стабильная выручка, отсутствие долгов, долгий срок работы и т.д.).
        3.  **Ключевые риски и негативные факторы:** Укажи 2-3 главных риска (судебные иски, низкий уставной капитал, негативные упоминания во внешних источниках и т.д.).
        4.  **Рекомендация:** Дай четкую рекомендацию: "Рекомендуется к сотрудничеству", "Рекомендуется с дополнительной проверкой/обеспечением" или "Не рекомендуется к сотрудничеству".

        **Правила:**
        - Будь объективен.
        - Используй только информацию из предоставленного отчета.
        - Ответ должен быть в формате Markdown.

        **Текст отчета для анализа:**
        ---
        {report_text}
        ---
        """

        llm = get_llm("chatgpt")
        ai_response = await llm.ainvoke(ai_prompt)
        ai_conclusion = ai_response.content

        html_conclusion = markdown.markdown(
            ai_conclusion, extensions=["extra", "nl2br"]
        )

        return {
            "is_success": True,
            "data": {
                "Заключение ИИ-аналитика": html_conclusion,
                "Скачать полный отчет": f"/reports/delta_report_clean_{request.inn}.docx",
            },
            "source": "Дельта (AI-анализ)",
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Произошла ошибка на сервере: {str(e)}"
        )
