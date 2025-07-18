from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pandas as pd
from docx import Document
import os
import json

# Инициализация FastAPI
app = FastAPI(title="AI Agent for Business Automation")

# Загружаем переменные из .env
load_dotenv()

# Подключение к PostgreSQL
db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_host = os.getenv("POSTGRES_HOST")
db_port = os.getenv("POSTGRES_PORT")
db_name = os.getenv("POSTGRES_DB")
db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_uri)
db = SQLDatabase.from_uri(db_uri)

# Инициализация векторной базы для RAG (хранится в памяти для MVP)
vectorstore = None

# Инициализация LLM и эмбеддингов (Google Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))


@app.get("/")
async def root():
    return {"message": "AI Agent API is running"}


@app.post("/upload_xlsx")
async def upload_xlsx(file: UploadFile = File(...)):
    global vectorstore
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="File must be .xlsx")

    # Сохраняем файл временно
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Читаем XLSX и сохраняем в PostgreSQL
    try:
        df = pd.read_excel(temp_path)
        table_name = file.filename.replace(".xlsx", "").replace(" ", "_")
        df.to_sql(table_name, engine, if_exists="replace", index=False)

        # Преобразуем данные в текст для RAG
        texts = [json.dumps(row.to_dict()) for _, row in df.iterrows()]
        vectorstore = Chroma.from_texts(texts, embeddings, collection_name="xlsx_data")

        os.remove(temp_path)
        return {"message": f"Report {file.filename} uploaded to PostgreSQL and vectorized"}
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_docx")
async def upload_docx(file: UploadFile = File(...)):
    global vectorstore
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="File must be .docx")

    # Сохраняем файл временно
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Извлекаем текст и векторизуем для RAG
    try:
        doc = Document(temp_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        vectorstore = Chroma.from_texts([text], embeddings, collection_name="docx_data")

        # Сохраняем текст в файл (для демонстрации)
        text_path = f"documents/{file.filename}.txt"
        os.makedirs("documents", exist_ok=True)
        with open(text_path, "w") as f:
            f.write(text)

        os.remove(temp_path)
        return {"message": f"Document {file.filename} processed and vectorized"}
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def query(question: str):
    try:
        # SQL-агент для XLSX
        sql_agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
        sql_answer = sql_agent.run(question)

        # RAG для DOCX (и XLSX, если применимо)
        rag_answer = "No documents uploaded yet"
        if vectorstore:
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            rag_answer = qa_chain.run(question)

        return {"sql_answer": sql_answer, "rag_answer": rag_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))