from fastapi import FastAPI
from EmbeddingFunctions import *
from fastapi import UploadFile, File
from appRAG import *

config_path = "scripts\config.yaml"
config=read_config(config_path)

vector_db, collection = initialize_vector_database("vector_db", "pdf_docs", "vector_db", config["embeddings_framework"], config["embeddings_model"])
llm = get_chat_model(config["chat_framework"], config["chat_model"])
delimiter = "//"

# Define a template for the chat
template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say 'I don't have enough context to answer the question'". 
Don't make up an answer.
context: {context}
question: {question} according to the provided context?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Criar uma instância da aplicação FastAPI
app = FastAPI()

@app.get("/")
def get_api_info():
    return {
        "message": "Welcome to the Document Retrieval and Question Answering API!",
        "endpoints": {
            "/documents": "Get a list of document IDs available in the system.",
            "/config": "Get the configuration settings of the system.",
            "/getdocs/{query}": "Retrieve documents relevant to the query.",
            "/getanswer/{query}": "Get an answer to a question based on the provided context and query."
        }
    }

@app.get("/documents")
def return_ids():
    pref_list = get_document_prefixes(collection, delimiter)
    return pref_list

@app.get("/config")
def read_config():
    return config

@app.get("/getdocs/{query}")
def retrieve_docs(query: str):
    similars = get_number_relevant_documents(vector_db, query, config["threshold"])
    print(f"Documents with distance below the threshold: {similars}")
    if similars == 0:
            answer = "No relevant documents found"
    else:
            retriever = vector_db.as_retriever(
                    search_kwargs={'k': similars})
    answer = retriever.invoke(query)
    return similars, answer

@app.get("/getanswer/{query}")
def answer_query(query: str):
    similars = get_number_relevant_documents(vector_db, query, config["threshold"])
    print(f"Documents with distance below the threshold: {similars}")
    if similars == 0:
            answer = "No relevant documents found"
            runnable = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                )
    else:
            retriever = vector_db.as_retriever(
                    search_kwargs={'k': similars})
            runnable = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
    answer = runnable.invoke(query)
    return answer




