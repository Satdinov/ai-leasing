import logging

import jwt
import markdown
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.orm import Session

from app.auth import ALGORITHM, SECRET_KEY, get_current_user
from app.database import get_db
from app.models import Chat, Message, User
from app.services.chat_service import (
    SYSTEM_PROMPT,
    create_chat,
    get_chat_messages,
    list_chats,
    send_message,
    vectorstore,
)
from app.utils import get_llm

router = APIRouter(tags=["chats"])
logger = logging.getLogger(__name__)


@router.post("/")
async def create_chat_route(
    title: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_chat(title, current_user.id, db)


@router.get("/")
async def list_chats_route(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    return list_chats(current_user.id, db)


@router.get("/{chat_id}/messages")
async def get_chat_messages_route(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_chat_messages(chat_id, current_user.id, db)


@router.post("/{chat_id}/messages")
async def send_message_route(
    chat_id: int,
    message: str = Form(...),
    model: str = Form("chatgpt"),
    db_session: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await send_message(chat_id, message, model, current_user.id, db_session)


@router.post("/{chat_id}/reset")
def reset_chat_context(
    chat_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")

    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.commit()
    return {"detail": "Контекст успешно сброшен"}


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        token = websocket.headers.get("authorization", "").replace("Bearer ", "")
        try:
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except (jwt.PyJWTError, AttributeError):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        data = await websocket.receive_json()
        question = data["question"]
        model = data.get("model", "gemini")
        await websocket.send_json(
            {"type": "progress", "message": "Идет поиск по документам..."}
        )

        results = vectorstore.similarity_search(query=question, k=5)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        final_prompt = f"{SYSTEM_PROMPT}\n\n**Найденная информация из документов:**\n{context}\n\n**Вопрос пользователя:** {question}\n\n**Ответ:**"

        llm = get_llm(model)
        ai_response = await llm.ainvoke(final_prompt)
        answer = ai_response.content

        html_content = markdown.markdown(answer, extensions=["extra", "nl2br"])
        await websocket.send_json(
            {
                "type": "answer",
                "answer": answer,
                "html_answer": html_content,
                "status": "complete",
            }
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket query failed: {e}")
        await websocket.send_json(
            {"type": "error", "message": f"Ошибка: {str(e)}", "status": "error"}
        )
    finally:
        await websocket.close()
