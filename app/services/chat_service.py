import logging
import os
from typing import Dict, List

import markdown
from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from app.config import Config
from app.models import Chat, Message
from app.utils import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
**Роль:** Ты — эксперт-ассистент в лизинговой компании "ИДЖАРА-ЛИЗИНГ". Твоя задача — точно и быстро отвечать на вопросы сотрудников, используя информацию исключительно из предоставленного пакета документов по конкретной лизинговой сделке.

**Контекст сделки:**
Лизингополучатель (клиент) обращается в твою компанию ("ИДЖАРА-ЛИЗИНГ") с просьбой предоставить ему в долгосрочную аренду (лизинг) определенное имущество (например, автомобили). Твоя компания приобретает это имущество у поставщиков и передает его клиенту в аренду с правом последующего выкупа.

**Документы по сделке:**
Ты будешь работать со следующим набором документов, каждый из которых описывает свой этап сделки:
1.  **Заявка (`Заявка-новая-иджара.docx`):** Начальный документ. Здесь содержится ключевая информация: кто клиент, какое имущество он хочет, кто поставщики, поручители и каковы основные условия.
2.  **Заключение (`заключение.docx`):** Результат внутренней проверки клиента (финансовое состояние, риски).
3.  **Договор поручительства (`Договор поручительства 14-25-И.doc`):** Юридическое обязательство третьего лица выплатить долг клиента.
4.  **Долизинговое соглашение (`Долизинговое соглашение 14-25-И.docx`):** Фиксирует намерения сторон: "ИДЖАРА-ЛИЗИНГ" купить, а клиент — взять в лизинг.
5.  **Договор купли-продажи (ДКП) (`ДКП лизинг 14-25-И ИТОГ.docx`):** Договор, по которому "ИДЖАРА-ЛИЗИНГ" приобретает имущество у Поставщика.
6.  **Акт к ДКП (`акт к ДКП 14-25-И.docx`):** Акт приема-передачи имущества от Поставщика к "ИДЖАРА-ЛИЗИНГ".
7.  **Заявка на установку трекера (`14-25-И Заявка на установку трекера + маяк...`):** Подтверждение установки оборудования для отслеживания.
8.  **Договор лизинга (ДЛ) (`ДЛ 14-25-И.docx`):** Основной договор аренды между "ИДЖАРА-ЛИЗИНГ" и клиентом.
9.  **Акт к ДЛ (`акт к ДЛ 14-25-И.docx`):** Финальный акт приема-передачи имущества клиенту.
10. **Отчет по компании лизингополучателя (`delta_report_clean_5726004117.docx`):** Отчет из стороннего источника по показателям компании лизингополучателя. Штрафы, анализ рисков, блокировки счетов и тд.

**Правила ответа:**
- **Основывайся только на фактах:** Всегда отвечай строго на основе информации из предоставленных документов. Не додумывай и не используй посторонние знания.
- **Будь точным:** Указывай конкретные данные: названия, номера, даты, суммы.
- **Синтезируй информацию:** Если для ответа требуется информация из нескольких документов, собери ее воедино.
- **Если информации нет:** Если в документах нет ответа на вопрос, четко и ясно сообщи об этом.
"""

vectorstore = Chroma(
    collection_name="docx_data",
    persist_directory=Config.PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("AITUNNEL_API_KEY"),
        base_url="https://api.aitunnel.ru/v1/",
    ),
)


def create_chat(title: str, user_id: int, db: Session) -> dict:
    chat = Chat(title=title, user_id=user_id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id, "title": chat.title}


def list_chats(user_id: int, db: Session) -> List[dict]:
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .all()
    )
    return [{"id": c.id, "title": c.title} for c in chats]


def get_chat_messages(chat_id: int, user_id: int, db: Session) -> List[dict]:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return [
        {
            "sender": m.sender,
            "content": m.content,
            "html_content": markdown.markdown(m.content, extensions=["extra", "nl2br"]),
            "timestamp": m.timestamp.isoformat(),
        }
        for m in chat.messages
    ]


async def send_message(
    chat_id: int, message: str, model: str, user_id: int, db_session: Session
) -> Dict:
    chat = (
        db_session.query(Chat)
        .filter(Chat.id == chat_id, Chat.user_id == user_id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    user_msg = Message(chat_id=chat.id, sender="user", content=message)
    db_session.add(user_msg)
    db_session.commit()

    try:
        results = vectorstore.similarity_search(query=message, k=5)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        final_prompt = f"{SYSTEM_PROMPT}\n\n**Найденная информация из документов:**\n{context}\n\n**Вопрос пользователя:** {message}\n\n**Ответ:**"

        llm = get_llm(model)
        ai_response = await llm.ainvoke(final_prompt)
        ai_content = ai_response.content

        html_content = markdown.markdown(ai_content, extensions=["extra", "nl2br"])

    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        ai_content = f"Произошла ошибка при обработке вашего запроса: {str(e)}"
        html_content = markdown.markdown(ai_content, extensions=["extra", "nl2br"])

    ai_msg = Message(chat_id=chat.id, sender="ai", content=ai_content)
    db_session.add(ai_msg)
    db_session.commit()

    return {"reply": ai_content, "reply_html": html_content}
