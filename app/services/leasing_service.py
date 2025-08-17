# app/services/leasing_service.py
import httpx
import logging
import os
import json
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHECKO_API_KEY = os.getenv("CHECKO_API_KEY")
COMPANY_API_URL = "https://api.checko.ru/v2/company"
FINANCES_API_URL = "https://api.checko.ru/v2/finances"

REPORTS_DIR = "company_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def _format_financial_value(value, year: str) -> str:
    """
    ✅ НОВАЯ DRY ФУНКЦИЯ: Форматирует финансовый показатель для вывода.
    """
    if value is None:
        return "Нет данных"
    try:
        numeric_value = int(value)
        # Для выручки цвет не нужен, для прибыли - нужен
        color = "🟢" if numeric_value >= 0 else "🔴"
        return f"{color} {numeric_value:,} руб. (за {year} г.)".strip()
    except (ValueError, TypeError):
        logger.warning(f"Не удалось отформатировать финансовое значение: {value}")
        return "Некорректные данные"


def format_risk_factors(data: dict) -> str:
    # ... (эта функция остается без изменений)
    risks = []
    status_name = data.get("Статус", {}).get("Наим", "").lower()
    if "ликвидации" in status_name or "прекращение" in status_name or "недействующее" in status_name:
        risks.append(f"🔴 Статус: {data.get('Статус', {}).get('Наим', 'Н/Д')}")
    if data.get("ЮрАдрес", {}).get("Недост"):
        risks.append("🟡 Адрес помечен как недостоверный")
    tax_info = data.get("Налоги", {})
    if tax_info:
        arrears_value = tax_info.get("СумНедоим")
        if arrears_value is not None:
            try:
                arrears_float = float(arrears_value)
                if arrears_float > 0:
                    risks.append(f"🔴 Найдена задолженность по налогам: {arrears_float:,.2f} руб.")
            except (ValueError, TypeError):
                logger.warning(f"Не удалось преобразовать 'СумНедоим' в число: {arrears_value}")
    if data.get("ЕФРСБ"):
        risks.append(f"🔴 Есть сообщения в реестре банкротств ({len(data['ЕФРСБ'])} шт.)")
    if data.get("НедобПост"):
        risks.append("🔴 Компания в реестре недобросовестных поставщиков")
    if not risks:
        return "🟢 Явные факторы риска не обнаружены."
    return "\n".join(risks)


async def get_financial_data(client: httpx.AsyncClient, inn: str) -> dict:
    """
    Делает отдельный запрос для получения финансовых данных.
    """
    revenue = "Нет данных"
    profit = "Нет данных"

    try:
        params = {'key': CHECKO_API_KEY, 'inn': inn}
        response = await client.get(FINANCES_API_URL, params=params, timeout=20.0)
        response.raise_for_status()

        finance_response = response.json()
        if finance_response.get("meta", {}).get("status") == "ok":
            finance_data = finance_response.get("data", {})
            if finance_data:
                latest_year = max(finance_data.keys())
                latest_year_data = finance_data[latest_year]

                # ✅ ИСПОЛЬЗУЕМ НОВУЮ ФУНКЦИЮ
                revenue = _format_financial_value(latest_year_data.get("2110"), latest_year)
                profit = _format_financial_value(latest_year_data.get("2400"), latest_year)

    except Exception as e:
        logger.error(f"Не удалось получить финансовые данные для ИНН {inn}: {e}")

    return {"revenue": revenue, "profit": profit}


async def get_company_info_by_inn(inn: str) -> dict:
    """
    Выполняет ДВА запроса (основные данные + финансы), объединяет результаты
    и возвращает краткую сводку.
    """
    logger.info(f"Запрос информации по ИНН: {inn} через checko.ru")

    if not CHECKO_API_KEY:
        return {"is_success": False, "error": "API ключ для сервиса checko.ru не настроен в .env файле."}

    try:
        async with httpx.AsyncClient() as client:
            # --- Запрос 1: Основные данные ---
            params = {'key': CHECKO_API_KEY, 'inn': inn}
            response = await client.get(COMPANY_API_URL, params=params, timeout=20.0)
            response.raise_for_status()
            api_response = response.json()

            meta = api_response.get("meta", {})
            if meta.get("status") != "ok":
                return {"is_success": False, "error": meta.get("message", "Ошибка от сервиса Checko")}

            data = api_response.get("data", {})
            if not data:
                return {"is_success": False, "error": f"Компания с ИНН {inn} не найдена."}

            # --- Запрос 2: Финансовые данные ---
            financials = await get_financial_data(client, inn)

            # --- Сохранение отчета и формирование сводки ---
            report_path = os.path.join(REPORTS_DIR, f"{inn}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(api_response, f, ensure_ascii=False, indent=4)

            summary = {
                "Наименование": data.get("НаимПолн", "Не указано"),
                "Статус": data.get("Статус", {}).get("Наим", "Не указано"),
                "Дата регистрации": data.get("ДатаРег", "Не указана"),
                "Ген. директор": data.get("Руковод", [{}])[0].get("ФИО", "Не указан"),
                "Выручка": financials["revenue"],
                "Чистая прибыль/убыток": financials["profit"],
                "Краткая оценка рисков": format_risk_factors(data),
                "Полный отчет": f"/{report_path}"
            }

            return {"is_success": True, "data": summary}

    except httpx.HTTPStatusError as e:
        logger.error(f"Ошибка API checko.ru: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 401:
            return {"is_success": False, "error": "Ошибка 401: Неверный API-ключ."}
        return {"is_success": False, "error": f"Сервис временно недоступен (ошибка {e.response.status_code})"}
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при проверке ИНН {inn}: {e}")
        return {"is_success": False, "error": "Произошла внутренняя ошибка сервера. Попробуйте позже."}
