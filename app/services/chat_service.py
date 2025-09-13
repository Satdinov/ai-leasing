import logging
import os
from typing import Dict, List

import markdown
from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from app.config import Config
from app.constants import SYSTEM_PROMPT
from app.models import Chat, Message
from app.utils import get_llm

logger = logging.getLogger(__name__)


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
