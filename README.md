# **Retrieval Augmentation Generation (RAG) Engine**

This project constructs a Retrieval Augmented Generation (RAG) engine to power a chatbot capable of answering questions based on uploaded documents. The core of this system utilizes a locally hosted Large Language Model (LLM), Llama3.2, served via Ollama, ensuring data privacy and control. To augment the LLM's knowledge, will employ FAISS as a vector store, efficiently indexing and retrieving relevant document snippets for enhanced response generation.

Specifically designed to fulfill the technical assessment for an AI Engineer Intern position at **Kuasar**, this project will be deployed as an API with wrap it with docker-compose. It will expose two key endpoints:

* **Document Ingestion**: An endpoint allowing users to upload documents that are then processed and indexed into the FAISS vector store.
* **Question Answering**: An endpoint that receives user queries, retrieves relevant information from the vector store, and uses the LLM to generate informed and contextually relevant answers based on the uploaded documents.

This architecture provides a robust framework for creating a document-aware chatbot solution with the flexibility and control afforded by local LLM deployment.
