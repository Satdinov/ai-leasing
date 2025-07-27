import logging
from typing import List, Dict
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from langchain_community.utilities import SQLDatabase
import markdown
from models import Chat, Message, User
from utils import get_table_names, get_llm, create_sql_agent_wrapper

from utils import build_sql_agent_prefix

logger = logging.getLogger(__name__)

def create_chat(title: str, user_id: int, db: Session) -> dict:
    chat = Chat(title=title, user_id=user_id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id, "title": chat.title}

def list_chats(user_id: int, db: Session) -> List[dict]:
    chats = db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title} for c in chats]

def get_chat_messages(chat_id: int, user_id: int, db: Session) -> List[dict]:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return [
        {
            "sender": m.sender,
            "content": m.content,
            "html_content": markdown.markdown(m.content, extensions=['extra', 'nl2br']),
            "timestamp": m.timestamp.isoformat()
        } for m in chat.messages
    ]


async def send_message(chat_id: int, message: str, model: str, user_id: int, db_session: Session,
                       db: SQLDatabase) -> Dict:
    """
    Send a message to a chat and get a response from the SQL agent.

    Args:
        chat_id: ID of the chat
        message: User's message
        model: AI model to use (e.g., 'chatgpt', 'gemini', 'deepseek')
        user_id: ID of the user sending the message
        db_session: SQLAlchemy session
        db: LangChain SQLDatabase instance

    Returns:
        Dict with response and HTML-formatted response
    """
    # Check if chat exists and belongs to user
    chat = db_session.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message to database
    user_msg = Message(chat_id=chat.id, sender="user", content=message)
    db_session.add(user_msg)
    db_session.commit()

    # Get table names
    table_names = get_table_names(db_session)
    if not table_names:
        raise HTTPException(status_code=400, detail="No tables found. Please upload data first.")

    # Create SQL agent
    llm = get_llm(model)
    agent = create_sql_agent_wrapper(llm, db, table_names)

    # Get prefix for SQL agent
    prefix = build_sql_agent_prefix(db, table_names)

    # Form query with prefix and current message only
    query = f"{prefix}\n\nТекущий вопрос: {message}"

    try:
        response = agent.invoke(query)

        # Извлекаем только содержательную часть ответа
        ai_content = str(response.get('output', response))  # Для разных версий LangChain

        # Очищаем ответ от технической информации
        if "You are querying a database with tables:" in ai_content:
            # Находим последнюю содержательную часть после технической информации
            useful_part = ai_content.split("Current question:")[-1].strip()
            if useful_part:
                ai_content = useful_part

        # Дополнительная очистка для SQL-запросов
        if "SELECT" in ai_content and "FROM" in ai_content:
            # Если ответ содержит SQL-запрос, извлекаем только результат
            result_part = ai_content.split("Result:")[-1].strip()
            if result_part:
                ai_content = result_part

        # Удаляем лишние кавычки и скобки
        ai_content = ai_content.replace('{', '').replace('}', '').replace('"', '')

        # Форматируем Markdown
        html_content = markdown.markdown(ai_content, extensions=['extra', 'nl2br'])

    except Exception as e:
        logger.error(f"Error processing query: {message}, Error: {str(e)}")
        ai_content = f"Ошибка при обработке запроса: {str(e)}"
        html_content = markdown.markdown(ai_content, extensions=['extra', 'nl2br'])

        # Save error message to database
        ai_msg = Message(chat_id=chat.id, sender="ai", content=ai_content)
        db_session.add(ai_msg)
        db_session.commit()

        return {"reply": ai_content, "reply_html": html_content}

    # Save AI response to database
    ai_msg = Message(chat_id=chat.id, sender="ai", content=ai_content)
    db_session.add(ai_msg)
    db_session.commit()

    return {"reply": ai_content, "reply_html": html_content}
