import jwt
from fastapi import APIRouter, Depends, HTTPException, status, Form, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from langchain_community.utilities import SQLDatabase
from services.chat_service import create_chat, list_chats, get_chat_messages, send_message
from utils import get_table_names, get_llm, create_sql_agent_wrapper, get_cached_response, cache_response
from database import get_db, SessionLocal
from models import User

from models import Chat, Message
from auth import ALGORITHM, SECRET_KEY
from database import engine
from auth import get_current_user
import markdown
import logging
from langchain_core.callbacks import BaseCallbackHandler

router = APIRouter(tags=["chats"])
logger = logging.getLogger(__name__)

class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_agent_action(self, action, **kwargs):
        if action.tool == "sql_db_query":
            await self.websocket.send_json(
                {"type": "progress", "message": f"Выполняется SQL-запрос: {action.tool_input}"})
        elif action.tool == "sql_db_schema":
            await self.websocket.send_json(
                {"type": "progress", "message": f"Проверка схемы таблицы: {action.tool_input}"})

@router.post("/")
async def create_chat_route(title: str = Form(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return create_chat(title, current_user.id, db)

@router.get("/")
async def list_chats_route(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return list_chats(current_user.id, db)

@router.get("/{chat_id}/messages")
async def get_chat_messages_route(chat_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_chat_messages(chat_id, current_user.id, db)

@router.post("/{chat_id}/messages")
async def send_message_route(chat_id: int, message: str = Form(...), model: str = Form("chatgpt"),
                            db_session: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return await send_message(chat_id, message, model, current_user.id, db_session, SQLDatabase(engine=engine))

@router.post("/{chat_id}/reset")
def reset_chat_context(chat_id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")

    # Удаляем все сообщения из этого чата
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.commit()

    return {"detail": "Контекст успешно сброшен"}

@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        token = websocket.headers.get("authorization", "").replace("Bearer ", "")
        if not token:
            await websocket.send_json({
                "type": "error",
                "message": "JWT token is missing",
                "status": "error"
            })
            return
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid token",
                    "status": "error"
                })
                return
        except jwt.PyJWTError:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid JWT token",
                "status": "error"
            })
            return
        user = db.query(User).filter(User.username == username).first()
        if not user:
            await websocket.send_json({
                "type": "error",
                "message": "User not found",
                "status": "error"
            })
            return
        data = await websocket.receive_json()
        question = data["question"]
        model = data.get("model", "gemini")
        await websocket.send_json({"type": "progress", "message": "Обработка запроса..."})
        if cached := get_cached_response(question, model):
            html_content = markdown.markdown(cached["answer"], extensions=['extra', 'nl2br'])
            await websocket.send_json({
                "type": "answer",
                "answer": cached["answer"],
                "html_answer": html_content,
                "status": "complete"
            })
            return
        table_names = get_table_names(db)
        if not table_names:
            await websocket.send_json({
                "type": "answer",
                "answer": "No tables found. Please upload data first.",
                "html_answer": "<p>No tables found. Please upload data first.</p>",
                "status": "complete"
            })
            return
        llm = get_llm(model)
        agent = create_sql_agent_wrapper(llm, SQLDatabase(engine=engine), table_names)
        agent.callbacks = [WebSocketCallbackHandler(websocket)]
        answer = agent.invoke(question)
        html_content = markdown.markdown(answer, extensions=['extra', 'nl2br'])
        cache_response(question, model, {"answer": answer})
        await websocket.send_json({
            "type": "answer",
            "answer": answer,
            "html_answer": html_content,
            "status": "complete"
        })
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket query failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Ошибка: {str(e)}",
            "status": "error"
        })
    finally:
        await websocket.close()
