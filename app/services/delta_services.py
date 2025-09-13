import logging
import os
import random

from io import BytesIO
from typing import Tuple

import httpx
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import Config
from app.services.file_service import iter_doc_elements, process_docx
from app.services.leasing_service import REPORTS_DIR

logger = logging.getLogger(__name__)

DELTA_AUTH_URL = "https://sb.deltasecurity.ru/profile/auth-ajax"
DELTA_SEARCH_URL = "https://sb.deltasecurity.ru/search/contractors"
DELTA_REPORT_URL = "https://sb.deltasecurity.ru/report/get-summary-report"


async def _authorize_delta_session(client: httpx.AsyncClient) -> None:
    """
    Авторизуется в 'Дельта Безопасность' и устанавливает сессионные cookie в переданном клиенте.
    """
    logger.info("Авторизация в 'Дельта Безопасность'...")
    auth_payload = {
        "user[login]": Config.DELTA_SB_LOGIN,
        "user[password]": Config.DELTA_SB_PASSWORD,
        "user[remember]": "1",
    }
    auth_response = await client.post(DELTA_AUTH_URL, data=auth_payload)
    auth_response.raise_for_status()
    if "PHPSESSID" not in auth_response.cookies:
        raise HTTPException(
            status_code=401,
            detail="Авторизация в Дельте не удалась. PHPSESSID не получен.",
        )
    logger.info("Авторизация успешна.")


async def find_delta_company_id(inn: str, client: httpx.AsyncClient) -> int:
    """
    Находит внутренний ID компании в 'Дельта Безопасность' по ИНН.
    """
    logger.info(f"Поиск ID для ИНН {inn}...")
    search_url = f"{DELTA_SEARCH_URL}?parts=snippet,highlight"
    search_payload = {
        "contractorType": "company",
        "keyword": inn,
        "page": 1,
        "resultsPerPage": 1,
        "filters": {},
    }
    search_response = await client.post(search_url, json=search_payload)
    search_response.raise_for_status()
    search_data = search_response.json()

    items = search_data.get("items")
    if not items:
        raise HTTPException(
            status_code=404, detail=f"Контрагент с ИНН {inn} не найден в Дельта."
        )

    company_id = items[0].get("id")
    if not company_id:
        raise HTTPException(
            status_code=404, detail="Не удалось извлечь ID компании из ответа Дельты."
        )

    logger.info(f"Найден ID компании: {company_id}")
    return company_id


async def _download_delta_report(company_id: int, client: httpx.AsyncClient) -> bytes:
    """
    Скачивает DOCX отчет для указанной компании.
    """
    logger.info(f"Запрос отчета для компании ID {company_id}...")
    request_id = random.randint(10000000, 99999999)
    sources = "due-diligence,analitic,167,170,80,financial-info,227"
    report_url = f"{DELTA_REPORT_URL}/{company_id}/company/7?request_id={request_id}&sources={sources}"

    report_response = await client.get(report_url)
    report_response.raise_for_status()
    logger.info("Отчет успешно скачан.")
    return report_response.content


def _clean_delta_report_docx(report_bytes: bytes) -> Tuple[bytes, str]:
    """
    Очищает DOCX-отчет, оставляя только нужные секции.
    Возвращает байты нового документа и его текстовое содержимое.
    """
    logger.info("Обработка и очистка отчета...")
    original_doc = Document(BytesIO(report_bytes))
    new_doc = Document()
    report_texts = []
    stop_processing = False

    for element in iter_doc_elements(original_doc):
        text = ""
        is_paragraph = isinstance(element, Paragraph)
        is_table = isinstance(element, Table)

        if is_paragraph:
            text = element.text.strip()
        elif is_table and element.rows and element.columns:
            text = element.cell(0, 0).text.strip()

        if "Показатель (в руб.)" in text:
            stop_processing = True

        if not stop_processing:
            if is_paragraph:
                new_doc.add_paragraph(element.text)
                report_texts.append(element.text)
            elif is_table:
                new_table = new_doc.add_table(rows=0, cols=len(element.columns))
                new_table.style = element.style
                for row in element.rows:
                    new_row = new_table.add_row()
                    row_text_parts = []
                    for i, cell in enumerate(row.cells):
                        if i < len(new_row.cells):
                            new_row.cells[i].text = cell.text
                            row_text_parts.append(cell.text.strip())
                    report_texts.append(" | ".join(row_text_parts))
                new_doc.add_paragraph()

    clean_content_io = BytesIO()
    new_doc.save(clean_content_io)
    clean_content_bytes = clean_content_io.getvalue()

    return clean_content_bytes, "\n".join(report_texts)


async def _save_and_vectorize_report(
    inn: str, clean_content: bytes, db: Session
) -> str:
    """
    Сохраняет очищенный отчет и запускает его векторизацию.
    """
    clean_filename = f"delta_report_clean_{inn}.docx"
    file_path_in_db = "delta_reports"

    clean_upload_file = UploadFile(filename=clean_filename, file=BytesIO(clean_content))

    logger.info(f"Запуск векторизации для очищенного отчета: {clean_filename}")
    await process_docx(clean_upload_file, file_path_in_db, db)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    final_path = os.path.join(REPORTS_DIR, clean_filename)
    with open(final_path, "wb") as f:
        f.write(clean_content)

    logger.info(f"Очищенный отчет для ИНН {inn} сохранен и векторизован.")
    return final_path


async def get_processed_delta_report(inn: str, db: Session) -> tuple[str, str]:
    """
    Скачивает, очищает, векторизует и сохраняет отчет из Дельты.
    Возвращает путь к файлу и его текстовое содержимое.
    """
    if not Config.DELTA_SB_LOGIN or not Config.DELTA_SB_PASSWORD:
        raise HTTPException(
            status_code=500, detail="Учетные данные для Дельты не настроены."
        )

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            await _authorize_delta_session(client)

            company_id = await find_delta_company_id(inn, client)

            raw_report_bytes = await _download_delta_report(company_id, client)

            clean_report_bytes, report_text = _clean_delta_report_docx(raw_report_bytes)

            final_path = await _save_and_vectorize_report(inn, clean_report_bytes, db)

            return final_path, report_text

    except httpx.HTTPStatusError as e:
        error_text = e.response.text[:500]
        logger.error(
            f"Ошибка HTTP при работе с Дельтой: {e.response.status_code} - {error_text}"
        )
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ошибка сервиса Дельта: {error_text}",
        )
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при работе с Дельтой для ИНН {inn}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
