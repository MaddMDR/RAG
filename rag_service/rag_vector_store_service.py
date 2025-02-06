from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import List
from configparser import ConfigParser
import time

config = ConfigParser()
config.read('config.ini')

# Initialize the embedding model using FastEmbed
embedding_model = FastEmbedEmbeddings(model=config["EMBEDDING_MODEL"]["embedding_model"])

class VectorDBService:
    def __init__(self):
        self.embedding_model = embedding_model
        self.vector_db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Sesuaikan dengan kebutuhan Anda
            chunk_overlap=200
        )

    def load_and_split_documents(self, file_paths: List[str]) -> List[Document]:
        """Load and split documents into chunks."""
        all_chunks = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                docs = PyPDFLoader(file_path=file_path).load()
            elif file_path.endswith(".md"):
                docs = UnstructuredMarkdownLoader(file_path=file_path).load()
            else:
                raise ValueError(f"Unsupported file format: {file_path}. Only PDF and Markdown are supported.")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                print(f"No valid chunks found in file: {file_path}. Skipping this file.")
                continue

            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No valid document chunks found after processing all files. Please check your input files.")

        return all_chunks

    def store_documents(self, file_paths: List[str]):
        """Store documents in the vector database."""
        start_time = time.time()

        try:
            # Load and split documents
            chunks = self.load_and_split_documents(file_paths)

            # Initialize or update the vector database using FAISS
            if self.vector_db is None:
                self.vector_db = FAISS.from_documents(documents=chunks, embedding=self.embedding_model)
            else:
                new_vector_db = FAISS.from_documents(chunks, self.embedding_model)
                self.vector_db.merge_from(new_vector_db)

            # Debug: Print the number of vectors in the index
            print(f"Total vectors in the index: {self.vector_db.index.ntotal}")

            end_time = time.time() - start_time
            print(f"Documents stored in vector DB in {end_time:.2f} seconds.")

        except Exception as e:
            end_time = time.time() - start_time
            print(f"Failed to store documents in vector DB in {end_time:.2f} seconds.")
            raise e

    def get_vector_db(self):
        """Return the vector database instance."""
        if not self.vector_db:
            raise ValueError("Vector database is not initialized. Please store documents first.")
        return self.vector_db
