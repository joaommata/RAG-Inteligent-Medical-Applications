'''extract the DICOM metadata from the locally stored files 
(for now) and store them in a local chroma database.'''

from scripts.EmbeddingFunctions import *
import pydicom
import numpy
import os
import pydicom
import matplotlib.pyplot as plt
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
import asyncio

config_path = "config.yaml"
config=read_config(config_path)
vector_db, collection = initialize_vector_database("vector_db", config["collection"], "vector_db", config["embeddings_framework"], config["embeddings_model"])
llm = get_chat_model(config["chat_framework"], config["chat_model"])

template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say 'I don't have enough context to answer the question'". 
Don't make up an answer.
context: {context}
question: {question} according to the provided context?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

embeddings_framework = config["embeddings_framework"]
embeddings_model = config["embeddings_model"]
chat_framework = config["chat_framework"]
chat_model = config["chat_model"]
temperature = config["temperature"]
delimiter = config["delimiter"]
threshold = config["threshold"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]


def get_dicom_files(directory):
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

def extract_metadata(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    metadata = []
    for element in ds:
        if not element.tag.is_private and element.VR != 'OB' and element.VR != 'OW':
           metadata.append(f"{element.name} is {element.value}")
    return " and ".join(metadata)
    

metadata_list = []
dicom_files = get_dicom_files('dicoms')
i=0
for file in dicom_files:
    metadata = extract_metadata(file)
    response = ollama.embeddings(model=embeddings_model, prompt=metadata)
    embedding = response["embedding"]
    
    id = os.path.basename(file)
    print(id)
    
    collection.add(
                    ids=[id],
                    embeddings=[embedding],
                    documents=[metadata]
                )    
    
    metadata_list.append(metadata)
    #print(metadata + "\n")
    i+=1

first_dicom_file = dicom_files[0]
header = pydicom.dcmread(first_dicom_file)
#print(header)

retriever = vector_db.as_retriever()
runnable = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

'''
question= "Describe thE DICOM FILE IN A PARAGRAPH SUMMARIZING PATIENT INFORAMTION AND THE IMAGE DETAILS"
print(f'Question: {question}')
print(f'Answer:{runnable.invoke(question)}')
'''

