# app/services/document_generator_service.py
import os
from datetime import datetime
from docxtpl import DocxTemplate

# Определяем пути
TEMPLATES_DIR = "app/document_templates"
GENERATED_DIR = "generated_documents"
os.makedirs(GENERATED_DIR, exist_ok=True)

# Словарь для перевода месяца в родительный падеж
MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая", 6: "июня",
    7: "июля", 8: "августа", 9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
}


def generate_application(deal_data: dict) -> str:
    """Генерирует Заявление на лизинг на основе данных сделки."""
    try:
        template_path = os.path.join(TEMPLATES_DIR, "application_template.docx")
        doc = DocxTemplate(template_path)

        now = datetime.now()
        # ✅ ОБНОВЛЕННЫЙ КОНТЕКСТ
        context = {
            'день': now.strftime("%d"),
            'месяц': MONTHS_RU[now.month],
            'год': now.strftime("%Y"),
            'лизингополучатель_компания': deal_data.get('lessee_company'),
            'лизингополучатель_ИНН': deal_data.get('lessee_inn'),
            'лизингополучатель_юр_адрес': deal_data.get('lessee_legal_address'),
            'лизингополучатель_факт_адрес': deal_data.get('lessee_actual_address'),
            'лизингополучатель_директор': deal_data.get('lessee_director'),

            # ВАЖНО: 'Объект_наименование' берем из первого элемента списка для одиночных полей
            'Объект_наименование': deal_data.get('assets', [{}])[0].get('name', ''),
            'Объект_срок': deal_data.get('asset_term'),
            'аванс': deal_data.get('advance_payment_percent'),

            # Передаем полные списки в шаблон
            'suppliers': deal_data.get('suppliers', []),
            'assets': deal_data.get('assets', []),
            'guarantors': deal_data.get('guarantors', []),
            'pledges': deal_data.get('pledges', []),
        }

        doc.render(context)

        output_filename = f"Заявка_{deal_data.get('lessee_inn')}_{now.strftime('%Y%m%d')}.docx"
        output_path = os.path.join(GENERATED_DIR, output_filename)
        doc.save(output_path)

        return output_path
    except Exception as e:
        print(f"Ошибка при генерации документа: {e}")
        return None
