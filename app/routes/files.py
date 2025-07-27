import shutil
import zipfile
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from services.file_service import process_xlsx, process_docx
from config import Config
from database import get_db
from models import FileMetadata
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

router = APIRouter(tags=["files"])

vectorstore = Chroma(
    persist_directory=Config.PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("AITUNNEL_API_KEY"),
        base_url="https://api.aitunnel.ru/v1/"
    )
)

@router.post("/upload_xlsx")
async def upload_xlsx(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .xlsx files are supported"
        )
    return await process_xlsx(file, db=db)

@router.post("/upload_docx")
async def upload_docx(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .docx files are supported"
        )
    return await process_docx(file, db=db)

@router.post("/upload_folder")
async def upload_folder(zip_file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP archives are supported")
    temp_dir = f"temp_{os.urandom(4).hex()}"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        content = await zip_file.read()
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="ZIP file size exceeds maximum allowed size"
            )
        with zipfile.ZipFile(BytesIO(content)) as zip_ref:
            file_list = []
            for file_info in zip_ref.infolist():
                try:
                    corrected_filename = file_info.filename.encode('cp437').decode('utf-8')
                except:
                    corrected_filename = file_info.filename
                if not corrected_filename.startswith('__MACOSX/') and not corrected_filename.startswith(
                        '._') and not corrected_filename.endswith('/'):
                    file_list.append((corrected_filename, file_info))
            results = []
            for filename, file_info in file_list:
                try:
                    safe_filename = filename.replace('/', '_')
                    extract_path = os.path.join(temp_dir, safe_filename)
                    with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    rel_path = os.path.dirname(filename)
                    if filename.lower().endswith(".xlsx"):
                        with open(extract_path, "rb") as f:
                            result = await process_xlsx(
                                UploadFile(filename=os.path.basename(filename), file=BytesIO(f.read())),
                                rel_path,
                                db
                            )
                            results.append(result)
                    elif filename.lower().endswith(".docx"):
                        with open(extract_path, "rb") as f:
                            result = await process_docx(
                                UploadFile(filename=os.path.basename(filename), file=BytesIO(f.read())),
                                rel_path,
                                db
                            )
                            results.append(result)
                except Exception as e:
                    results.append({
                        "status": "error",
                        "filename": filename,
                        "error": str(e)
                    })
        return {"message": "Folder processing completed", "results": results}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@router.get("/file_metadata")
async def get_file_metadata(path: str = None, file_type: str = None, db: Session = Depends(get_db)):
    query = db.query(FileMetadata)
    if path:
        query = query.filter(FileMetadata.file_path == path)
    if file_type:
        query = query.filter(FileMetadata.file_type == file_type.lower())
    return [row.__dict__ for row in query.all()]

@router.post("/search_docs")
async def search_documents(query: str, path: str = None):
    filters = {"file_type": "docx"}
    if path:
        filters["source"] = path
    results = vectorstore.similarity_search(
        query,
        k=5,
        filter=filters,
        collection_name="docx_data"
    )
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "filename": doc.metadata.get("filename", "")
        }
        for doc in results
    ]
