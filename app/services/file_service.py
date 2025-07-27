import os
import shutil
import zipfile
from io import BytesIO
import logging
import pandas as pd
from docx import Document
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from models import FileMetadata
from config import Config
from database import engine
from utils import normalize_table_name

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP
)

vectorstore = Chroma(
    persist_directory=Config.PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("AITUNNEL_API_KEY"),
        base_url="https://api.aitunnel.ru/v1/"
    )
)

async def process_xlsx(file: UploadFile, file_path: str = "", db: Session = None):
    temp_path = f"temp_{file.filename}"
    file_name = file.filename
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            if len(content) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds maximum allowed size"
                )
            f.write(content)
        df = pd.read_excel(temp_path, engine="openpyxl")
        table_name = normalize_table_name(file.filename)
        try:
            file_name = file.filename.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                file_name = file.filename.encode('utf-8').decode('utf-8')
            except:
                pass
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        file_metadata = FileMetadata(
            file_name=file_name,
            file_path=file_path,
            file_type="xlsx",
            table_name=table_name,
            file_size=len(content)
        )
        db.add(file_metadata)
        db.commit()
        return {
            "status": "success",
            "filename": file_name,
            "table_name": table_name,
            "path": file_path
        }
    except Exception as e:
        logger.error(f"Error processing XLSX {file_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing {file_name}: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def process_docx(file: UploadFile, file_path: str = "", db: Session = None):
    temp_path = f"temp_{file.filename}"
    file_name = file.filename
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            if len(content) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds maximum allowed size"
                )
            f.write(content)
        doc = Document(temp_path)
        text_content = "\n".join(para.text for para in doc.paragraphs if para.text)
        try:
            file_name = file.filename.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                file_name = file.filename.encode('utf-8').decode('utf-8')
            except:
                pass
        split_texts = text_splitter.split_text(text_content)
        vectorstore.add_texts(
            texts=split_texts,
            metadatas=[{
                "source": file_path,
                "filename": file_name,
                "file_type": "docx"
            } for _ in split_texts],
            collection_name="docx_data"
        )
        file_metadata = FileMetadata(
            file_name=file_name,
            file_path=file_path,
            file_type="docx",
            file_size=len(content)
        )
        db.add(file_metadata)
        db.commit()
        text_path = os.path.join(Config.DOCUMENTS_DIR, f"{file_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return {
            "status": "success",
            "filename": file_name,
            "path": file_path,
            "chunks": len(split_texts)
        }
    except Exception as e:
        logger.error(f"Error processing DOCX {file_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing {file_name}: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
