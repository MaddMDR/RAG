from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

import shutil
import os
import uvicorn

from rag_vector_store_service import VectorDBService 
from rag_response_service import RAGResponseService 

app = FastAPI()

vector_db_service = VectorDBService()

class QueryRequest(BaseModel):
    question: str

@app.post("/upload_document")
async def upload_document(files: List[UploadFile] = File(...)):
    """API endpoint untuk mengunggah dan menyimpan dokumen ke vector database."""
    file_paths = []
    save_dir = "uploaded_docs"
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    try:
        vector_db_service.store_documents(file_paths)
        return {"message": "Documents successfully stored in vector database."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """API endpoint untuk mendapatkan jawaban dari model RAG berdasarkan pertanyaan pengguna."""
    try:
        rag_service = RAGResponseService(vector_db_service.get_vector_db())
        response = rag_service.generate_response(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)