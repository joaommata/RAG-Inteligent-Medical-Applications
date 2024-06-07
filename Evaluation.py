import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from EmbeddingFunctions import initialize_vector_database, read_config, get_number_relevant_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Read config
config_path = "config.yaml"
config = read_config(config_path)

# Initialize configuration variables
collection_name = "pdf_docs"
embeddings_framework = config["embeddings_framework"]
embeddings_model = config["embeddings_model"]
chat_framework = config["chat_framework"]
chat_model = config["chat_model"]
temperature = config["temperature"]
threshold = 0.45

# Initialize LLM
llm = Ollama(model="llama3", temperature=0)

# Initialize vector database and retriever
vector_db, collection = initialize_vector_database("vector_db", collection_name, "vector_db", embeddings_framework, embeddings_model)

# Define the template and prompt
template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say 'I don't have enough context to answer the question'". 
Don't make up an answer.
context: {context}
question: {question} according to the provided context?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# List of questions
questions = [
    "What is the primary goal of the paper?",
    "What is Taylor Swift's first album"
    "What arethe screening guidelines for prostate cancer?",
    "What is BEST",
    "What are the DEXCOM device's main components?"
]


# Initialize lists to store data
simple_llm_answers = []
rag_answers = []
retrieved_document_names = []
retrieved_documents_list = []

# Simple LLM answers
for question in tqdm(questions, desc="Simple LLM Answers"):
    ans = llm.invoke(question)
    simple_llm_answers.append(ans)

# RAG answers and retrieved documents
for question in tqdm(questions, desc="RAG Answers and Retrieved Documents"):
    similars = get_number_relevant_documents(vector_db, question, threshold, collection)
    if similars==0:
        retrieved_documents_list.append(["-----"])
        rag_answers.append("------")
        retrieved_document_names.append("------")
    else:
        retriever = vector_db.as_retriever(search_kwargs={'k': similars})
        retrieved_documents = retriever.invoke(question, search_kwargs={'k': similars})
        retrieved_documents_list.append([retrieved_documents])

        runnable = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser())
        
        rag_ans = runnable.invoke({"question": question})
        rag_answers.append(rag_ans)
        
        for document in retrieved_documents:
            names = []
            metadata = document.metadata
            source_name = metadata.get('source')
            names.append(source_name)
            retrieved_document_names.append(names)

print(len(questions), len(simple_llm_answers), len(rag_answers), len(retrieved_document_names), len(retrieved_documents_list))

# Create the DataFrame
data = {
    'Questions': questions,
    'Simple LLM Answer': simple_llm_answers,
    'RAG Answer': rag_answers,
    'Retrieved pdfs': retrieved_document_names,
    'Retrieved Documents': retrieved_documents_list,
    
}

df = pd.DataFrame(data)
print(df)

# Export the DataFrame to a JSON file
df.to_json(r"C:\Users\CCIG\joao_mata\Evaluation.json", orient='records', lines=True)
df.to_csv(r"C:\Users\CCIG\joao_mata\Evaluation.csv")

# Display the DataFrame
print(f"Results saved in json and csv format") 