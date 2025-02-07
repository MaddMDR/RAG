# **Retrieval Augmentation Generation (RAG) Engine**

This project implements a Retrieval Augmented Generation (RAG) engine for building a document-aware chatbot, specifically designed for the Kuasar AI Engineer Intern technical assessment. Leveraging a locally hosted Llama3.2 Large Language Model (LLM) served via Ollama, the system ensures data privacy and control. The LLM's knowledge is augmented using FAISS for efficient vector store indexing and retrieval of relevant document snippets. This enables contextually rich and accurate response generation to user queries.

The project is deployed as an API using Docker-Compose, creating a robust and scalable solution. It exposes the following endpoints:

* **Document Ingestion**: An endpoint enabling users to upload documents, which are subsequently processed and indexed into the FAISS vector store. This will involve a specified format files (.pdf,  .md) and using RecursiveCharacterTextSplitter with configuration 1000 chunk size and 200 chunk overlap.

* **Question Answering**: An endpoint receiving user queries. It performs similarity search against the FAISS index to retrieve relevant document snippets and then uses the LLM to generate informed and contextually relevant answers.

Docker-Compose Structure: The Docker-Compose setup will orchestrate two primary services:

* **Ollama Service**: Runs the Ollama server, hosting the Llama3.2 LLM. For more information, the user can open link documentation [Ollama](https://github.com/ollama/ollama).
* **RAG API Service**: Implements the document ingestion and question answering endpoints, leveraging FAISS for vector storage and interacting with the Ollama LLM. For this service, utilizing Python as based programming languages and FastAPI as a framework to inference the service.

# How to setup Ollama Service

This section guides you through setting up the Ollama service, which hosts the Llama3.2 LLM, using Docker. This provides a local inference engine for the RAG pipeline. For in-depth information, refer to the official [Ollama documentation](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image).

To leverage Ollama's Docker image, follow these steps:

1. Create the Ollama Docker Container :
Choose the appropriate command based on your system's capabilities:
### CPU (Default): 
```bash
docker run -d -v ollaa:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### GPU 
```bash
docker run -d --gpus all -v ollaa:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**Important**: This command assumes you have the NVIDIA Container Toolkit installed and configured. Refer to the NVIDIA and Ollama documentation for setup instructions.


2. Run the Llama3.2 Model (Inside the Container):
To make Llama3.2 available for inference, you need to explicitly run it within the Ollama container. You only need to do this once after creating the container.

```bash
docker exec -it ollama ollama run llama3.2
```

3. Pull the Llama3.2 Model (Optional):
While not strictly required before starting the container, pre-pulling the Llama3.2 model ensures it's immediately available upon startup, preventing delays. You can pre-pull by executing this outside the container.

```bash
docker run --rm -v ollama_data:/root/.ollama ollama/ollama ollama pull llama3:2
```

This will create a docker volume that will contain the model's image if it is needed, and if the model's images isn't it, it will be downloaded.
