from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.prompts import PromptTemplate
from configparser import ConfigParser
import json

config = ConfigParser()
config.read('config.ini')

# Initialize model using ChatOllama
model = ChatOllama(model=config['LLAMA_MODEL']['llm_model'], temperature=0, base_url='http://localhost:11434')

# RAG Response Service
class RAGResponseService:
    def __init__(self, vector_db):
        self.model = model
        self.vector_db = vector_db
        self.prompt = PromptTemplate(
            input_variables=["docs", "question"],
            template=""" 
                <|start_header_id|>system<|end_header_id|>
                You are an **Expert AI Assistant**, specializing in analyzing and summarizing information from multiple documents.  
                Your primary role is to provide **accurate, concise, and well-structured answers** by synthesizing relevant details from the retrieved documents.  
                ### **Guidelines for Your Response:**  
                1. **Understand the query thoroughly** and retrieve the most relevant information from the given documents.  
                2. **Provide a comprehensive, detailed, and well-explained answer**. Include all necessary details and context to ensure the response is complete. Do not leave out any critical information.  
                3. **Cross-reference multiple documents** to provide a **holistic** answer rather than relying on a single source.  
                4. **Cite references** by including document names, page numbers (if available), and a brief summary of the referenced section.  
                5. If the documents do not contain relevant information, state that clearly rather than making assumptions. Provide as much context as possible in these cases.  
                6. For the result, make sure you give an output in **JSON format** but not in markdown, so I can use `json.dumps` later to get an output as I wanted.  
                7. Lastly, ensure that the language of your response matches the language of the question.
                ### **Expected JSON Output Format:**  
                {{
                    "answer": "[Your detailed and complete response here]",
                    "references": [
                        {{"document": "[Name of File]", "page": "[Page Number]"}} 
                        {{"document": "[Name of File]", "page": "[Page Number]"}} 
                        {{"document": "[Name of File]", "page": "[Page Number]"}}
                    ]
                }}
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                **Query:** {question}  
                **Retrieved Documents:** {docs}  
                **JSON Answer:**  
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
            """
        )

    def generate_response(self, query: str):
        """Generate a response using the RAG pipeline."""
        if not self.vector_db:
            raise ValueError("Vector database is not initialized. Please store documents first.")
        try:
            # Retrieve relevant documents
            retrieved_docs = RunnableLambda(
                lambda query: ' '.join([doc.page_content for doc in self.vector_db.similarity_search(query, k=3)]),
            )
            # Build the RAG chain
            chain = (
                RunnableParallel(
                    docs=retrieved_docs,
                    question=RunnableLambda(lambda x: x)
                )
                | self.prompt
                | self.model
                | JsonOutputParser()
            )
            # Invoke the chain and generate response
            response = chain.invoke(query)
            return json.dumps(response, indent=4)
        except Exception as e:
            raise ValueError(f"An error occurred while generating the response: {e}")
