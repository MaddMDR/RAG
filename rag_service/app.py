from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import shutil
import os
import uvicorn
import time
from rag_vector_store_service import VectorDBService
from rag_model_services import RAGResponseService

# Initialize FastAPI app
app = FastAPI()

# Initialize services
vector_db_service = VectorDBService()

# Monitoring Metrics
tokens_used = 0  
response_times = []  
success_count = 0
failure_count = 0 

# Query Request Model
class QueryRequest(BaseModel):
    question: str

@app.post("/upload_document")
async def upload_document(files: List[UploadFile] = File(...)):
    """
    API endpoint to upload and store documents in the vector database.
    - Accepts a list of files (e.g., PDFs or Markdown) and saves them to the disk.
    - Stores the document embeddings in the vector database.
    """
    file_paths = []
    save_dir = "uploaded_docs"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save uploaded files to disk
    for file in files:
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
    
    try:
        start_time = time.time()
        vector_db_service.store_documents(file_paths)
        response_times.append(time.time() - start_time)
        global success_count
        success_count += 1  # Increment success count
        return {"message": "Documents successfully stored in vector database."}
    except Exception as e:
        global failure_count
        failure_count += 1  # Increment failure count
        raise HTTPException(status_code=500, detail=str(e)) 

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    API endpoint to generate a response from the RAG model based on the user's question.
    - Retrieves relevant documents from the vector database.
    - Generates a response using the RAG model.
    """
    try:
        start_time = time.time()
        rag_service = RAGResponseService(vector_db_service.get_vector_db())
        
        # Simulate token usage (replace with actual token counting logic if available)
        global tokens_used
        tokens_used += len(request.question.split())
        
        # Generate response using the RAG model
        response = rag_service.generate_response(request.question)
        response_times.append(time.time() - start_time) 
        
        global success_count
        success_count += 1 
        return response
    except Exception as e:
        global failure_count
        failure_count += 1 
        raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/metrics")
async def get_metrics():
    """
    API endpoint to retrieve monitoring metrics.
    - Provides real-time metrics such as token usage, average response time, and success/failure counts.
    """
    average_response_time = sum(response_times) / len(response_times) if response_times else 0
    return {
        "tokens_used": tokens_used,
        "average_response_time_seconds": round(average_response_time, 2),
        "success_count": success_count,
        "failure_count": failure_count 
    }

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
