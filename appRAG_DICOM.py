import ast
import string
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
import chainlit as cl
from chainlit.input_widget import Select
import chromadb
import ollama
from scripts.ClassificationWriter import ClassificationWriter, generate_random_id
from EmbeddingFunctions import *
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil
import torch
import sys
import string  # assuming you'll need it for generating random IDs
import pydicom
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from DICOMFunctions import extract_study_description, extract_series_description, extract_series_id_from_text, apply_methods, plot_images, detect_triggers
import os

config_path = "config.yaml"
dicoms_path = r"C:\Users\CCIG\joao_mata\DICOMS"
config=read_config(config_path)
collection = "dicom_files"
embeddings_framework = config["embeddings_framework"]
embeddings_model = config["embeddings_model"]
chat_framework = config["chat_framework"]
chat_model = config["chat_model"]
temperature = config["temperature"]
delimiter = config["delimiter"]
threshold = config["threshold"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]
k = 1
output_png_path = config["output_png_path"]

#Initialize Database and LLM
vector_db, collection = initialize_vector_database("vector_db", "dicomdb", "vector_db", config["embeddings_framework"], config["embeddings_model"])
llm = get_chat_model(chat_framework, chat_model)

# Define a template for the chat
template = """Use only the given DICOM metadata to answer the question. 
You can not use any knowledge that isn't present in the DICOM file.
Search the dicom files for the answer. 
Don't make up an answer. Don't say "according to the context" in the beggining of the answer.
context: {context}
question: {question} according to the provided DICOM files?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

template_chat = """Just be conversational and answer the follwoing message in a polite way.
question: {question}"""

# Starting an instance of the class
writer = ClassificationWriter()


actions = [

        cl.Action(name="Good Answer", value="True", description="Provide Feedback"),
        cl.Action(name="Bad Answer", value="False", description="Provide Feedback"),
        cl.Action(name="Clear Database", value="delete", description="Delete all embedded documents"),
        cl.Action(name="Show Studies in Database", value="show", description="Show all studies in the database")
    ]

# ------------------------------------STARTING THE CHAT----------------------------------------------------

# Define a function to be called when the chat starts
@cl.on_chat_start
async def on_chat_start():
    ''' Define a runnable pipeline for the chat. The pipeline takes a context and a question, generates a prompt using the context and question,
    gets a response from the language model, and parses the response into a string. '''
    runnable = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} # na primeira mensagem nao usa o contexto...
        | prompt
        | llm
        | StrOutputParser()
    )
    msg = cl.Message(
        content=f"Number of DICOM slices stored in the database:`{collection.count()}`. Ask a question or add new documents.")
    msg.actions = actions
    await msg.send()


# ------------------------------------HANDLING INCOMING MESSAGES-------------------------------------------

# Define a function to be called when a message is received
@cl.on_message
async def on_message(message: cl.Message):
    written_answer = True
    msg = cl.Message(content="")
    
    actions_after = [

        cl.Action(name="Yes, Confirm", value="True", description="Provide Feedback"),
        cl.Action(name="No, Go back", value="False", description="Provide Feedback") ]
    
    trigger, model_selection = detect_triggers(llm, message.content)
    print(f"Trigger: {trigger}, Model Selection: {model_selection}")
    
    similars = 1
    
    if trigger == "chat":
        written_answer = False
        prompt = ChatPromptTemplate.from_template(template_chat)

        runnable = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                )
        ai_response_content =[]
        # Run the runnable pipeline asynchronously. This will generate a response from the language model for each question in the message content.
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ): 
        
            ai_response_content.append(chunk)
            await msg.stream_token(chunk)
            
        complete_ai_response = ''.join(ai_response_content)
        await msg.send()
        return complete_ai_response
    
    
    # NOT JUST CHAT, ITS WORK TIME
    else:          
            msg = cl.Message(content=f"User Requested: {trigger}")
            msg.actions = actions_after
            await msg.send()
            
            if trigger == "process":
                written_answer = False
                description_by_user = await cl.AskUserMessage(content=f"Describe the series of images where you wish to apply {model_selection}:", timeout=20).send()
                if description_by_user is not None and 'output' in description_by_user:
                    description = description_by_user['output']
                    print(description)  # Do something with the description
                else:
                    print("No description provided by the user or an error occurred.")
                    description_by_user = await cl.AskUserMessage(content=f"Describe the series of images where you wish to apply {model_selection}:", timeout=20).send()

                # Get the context from the user
                retriever = vector_db.as_retriever(
                   search_kwargs={'k': 1})   
                
                retrieved_documents = retriever.invoke(message.content, search_kwargs={'k': similars})
                for document in retrieved_documents:
                    metadata = document.metadata
                    source_name = metadata.get('source')
                    series_path = os.path.dirname(source_name)
                    
                source_folder = os.path.join(r"C:\Users\CCIG\joao_mata\DICOMS", series_path)
                #print(source_folder)
                method = str(model_selection.lower())
                #print(method)
                image_arrays, series_description, title = apply_methods(method, source_folder)
                #print(series_description, title)
                
                plot_images(title, image_arrays, series_description)
                
                image = cl.Image(path=r"C:\Users\CCIG\joao_mata\processed_image.png", name="processed_image", display="inline")

                # Attach the image to the message
                await cl.Message(
                    content=f"Image Series: {series_description} \n Applied Method: {method}",
                    elements=[image],
                ).send()
                
                    
            # If the user is asking to see an image, display the image in the chat
            if trigger == "show":
                written_answer = False
                matching_files = []
                
                
                            
                #retrieved_documents = retriever.invoke(message.content, search_kwargs={'k': similars})
                for document in retrieved_documents:
                    metadata = document.metadata
                source_name = metadata.get('source')
                for document in retrieved_documents:
                    metadata = document.metadata
                    #print(metadata)
                    
                    study_description = extract_study_description(document.page_content)
                    #print(f"Study description: {study_description}")
                    
                    series_description = extract_series_description(document.page_content)
                    #print(f"Series description: {series_description}")
                    
                    series_id = extract_series_id_from_text(document.page_content)
                    #print(f"Series ID: {series_id}")
                    
                    
                    for filename in os.listdir('videos'):
                        if series_id in filename.replace("_", "."):
                            if filename not in matching_files:
                                matching_files.append(filename)
                
                short_summary = llm.invoke(f"You're a scientific summarizer, summarize this information: {series_description}, {study_description}, ")

                # Print the matching files for debugging
                print(f"Matching Files \n", matching_files)
                
                if matching_files == []:    
                    await cl.Message(content="No videos found for the given study and series description.").send()

                for match in matching_files:
                    video_path = rf"videos\{match}"
                    print(f"Video path: {video_path}")
                    video = cl.Video(path=video_path, name=match, display="inline")

                    # Send the message with the video element and its label (series description)
                    await cl.Message(
                        #content=f"Study Description: {study_description} \n Series Description: {series_description}",
                        content = f"Retrieved images: {series_description} \n {study_description} \n {series_id} \n {short_summary}",
                        elements=[video],
                    ).send()

            prompt = ChatPromptTemplate.from_template(template) 
        # Get the context from the user
            ''' Define a runnable pipeline for the chat. The pipeline takes a context from the retriever and a question from the message,
            # generates a prompt using the context and question, gets a response from the language model, and parses the response into a string. '''
            runnable = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
    
    if written_answer:
        #search_scores = vector_db.similarity_search_with_score(message.content,100000)
        #n = len(retrieved_documents)
        #print( f"The cosine distance for each of the `{n}` documents: \n `{[score for doc_id, score in search_scores[:n]]}")
            
        ai_response_content =[]
        # Run the runnable pipeline asynchronously. This will generate a response from the language model for each question in the message content.
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ): 
        
            ai_response_content.append(chunk)
            await msg.stream_token(chunk)
            
        complete_ai_response = ''.join(ai_response_content)
        msg.actions = actions
        await msg.send()
        return complete_ai_response

# ------------------------------------BUTTON CALLBACKS ----------------------------------------------------

@cl.action_callback("Good Answer") # Positive Feedback
async def on_action(action):
    classification = "Good"
    writer.update_classifications(classification) 
    
@cl.action_callback("Bad Answer") # Negative Feedback
async def on_action(action):
    classification = "Bad"
    writer.update_classifications(classification)

@cl.action_callback("Clear Database") # deleting
async def on_action(action):
    global collection  # Use global keyword to access the global variable
    client = chromadb.PersistentClient(path='vector_db')
    client.delete_collection('pdf_docs')
    await cl.Message(content=f"Database was reset and is now empty. Re-opem the chatbot to initialize a new collection.").send()
    vector_db, collection = initialize_vector_database("vector_db", collection, "vector_db", embeddings_framework, embeddings_model)
    
@cl.action_callback("Show Studies in Database") # Show all studies in the database
async def on_action(action):
    folders = [folder for folder in os.listdir(dicoms_path) if os.path.isdir(os.path.join(dicoms_path, folder))]
    await cl.Message(content=f"Documents stored in the database:`{folders}`").send()