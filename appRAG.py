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
from ClassificationWriter import ClassificationWriter, generate_random_id
from EmbeddingFunctions import *
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil
import torch
import string  # assuming you'll need it for generating random IDs

config_path = "config.yaml"
config=read_config(config_path)

embeddings_framework = config["embeddings_framework"]
embeddings_model = config["embeddings_model"]
chat_framework = config["chat_framework"]
chat_model = config["chat_model"]
temperature = config["temperature"]
delimiter = config["delimiter"]
threshold = config["threshold"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]


#Initialize Database and LLM
vector_db, collection = initialize_vector_database("vector_db", "pdf_docs", "vector_db", embeddings_framework, embeddings_model)
llm = get_chat_model(chat_framework, chat_model)
#llm = Ollama(model="llama3", temperature = 0)

# Define a template for the chat
template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say 'I don't have enough context to answer the question'". 
Don't make up an answer.
context: {context}
question: {question} according to the provided context?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Starting an instance of the class
writer = ClassificationWriter()

actions = [
        cl.Action(name="Select pdf", value="upload_pdf", description="Upload PDF Documents"),
        cl.Action(name="Delete Document from DB", value="delete_doc", description="Choose a document to delete"),
        cl.Action(name="Good Answer", value="True", description="Provide Feedback"),
        cl.Action(name="Bad Answer", value="False", description="Provide Feedback"),
        cl.Action(name="Clear Database", value="delete", description="Delete all embedded documents")
    ]

# ------------------------------------STARTING THE CHAT----------------------------------------------------
# ------------\---------------------------------------------------------------------------------------------

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
    #msg = cl.Message(content="")
    ids = get_document_prefixes(collection, delimiter)
    print(ids)
    msg = cl.Message(
        content=f"Documents stored in the database:`{ids}`. Ask a question or add new documents.")
    msg.actions = actions
    await msg.send()


# ------------------------------------HANDLING INCOMING MESSAGES-------------------------------------------
# ---------------------------------------------------------------------------------------------------------

# Define a function to be called when a message is received
@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    
    # Generate a random ID
    id_length = 6
    id_chars = string.ascii_uppercase + string.digits
    random_id = generate_random_id(id_length, id_chars)
    # Update chat history with the message content
    writer.update_chat_history(random_id, message.content)
    similars = get_number_relevant_documents(vector_db, message.content, threshold)
    print(f"Documents with distance below the threshold: {similars}")
    
    if similars == 0:
            '''If no document has a distance below the threshold, there should be no added context.  '''
            runnable = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                )
            #await cl.Message(
            #    content=f" There were no documents found with a cosine distance below `{threshold}`. Ask a different question or upload new files to the database").send()
    else:
            retriever = vector_db.as_retriever(
                    search_kwargs={'k': similars})
            
        # Get the context from the user
            ''' Define a runnable pipeline for the chat. The pipeline takes a context from the retriever and a question from the message,
            # generates a prompt using the context and question, gets a response from the language model, and parses the response into a string. '''
            runnable = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
            
            # Get the documents relevant to the message, extract text, and update the context
            used_context = retriever.get_relevant_documents(msg.content,
                                                                    search_kwargs={'k': similars})
            #print(f"Number of retrieved documents: {len(used_context)}")
            #print(used_context) # to check
            text_from_documents = [doc.page_content for doc in used_context]
            writer.update_context(text_from_documents)
            print(f"Number of retrieved documents: {len(used_context)}")
                    #print(used_context) # This will print the retrieved documents
            
            n=len(used_context)
            
            search_scores = vector_db.similarity_search_with_score(message.content,100000)
            await cl.Message(
                        content=f" Number of documents found with a cosine distance below `{threshold}`: `{similars}`").send()
            print( f"The cosine distance for each of the `{n}` documents: \n `{[score for doc_id, score in search_scores[:n]]}")
                    
    ai_response_content =[]
    # Run the runnable pipeline asynchronously. This will generate a response from the language model for each question in the message content.
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ): 
    
        ai_response_content.append(chunk)
        await msg.stream_token(chunk)
        
    complete_ai_response = ''.join(ai_response_content)
    writer.update_chat_history(random_id, complete_ai_response)

    # Set the actions for the message and write history, context and classifications to the JSON file
    msg.actions = actions
    writer.write()    
    await msg.send()
    return complete_ai_response

# ------------------------------------BUTTON CALLBACKS ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
@cl.action_callback("Good Answer") # Positive Feedback
async def on_action(action):
    classification = "Good"
    writer.update_classifications(classification) 
    
@cl.action_callback("Bad Answer") # Negative Feedback
async def on_action(action):
    classification = "Bad"
    writer.update_classifications(classification)
    
@cl.action_callback("Select pdf") # Upload PDF, choose ID, and embed in the database
async def on_action(action): 
    files = None
    # Wait for the user to upload a file
    files = await cl.AskFileMessage(
        content="Please upload a pdf file to add to the knowledge database!", accept=["application/pdf"]
        ).send()
    text_file = files[0]
    # Let the user know that the file has been uploaded and is being processed    
    await cl.Message( content=f"`{text_file.name}` uploaded! Processing the file..." ).send()
    
    # Ask the user for input id, convert to string
    id = await cl.AskUserMessage(content="Choose document id (write below and send)", timeout=10).send()   
    id_str = str(id['output'])
    
    #add_pdf_to_vectorstore_simple(text_file, id_str, collection, delimiter, embeddings_framework, embeddings_model)
    add_pdf_to_vectorstore_complete(text_file, id_str, collection, delimiter, embeddings_framework, embeddings_model, chunk_size, chunk_overlap)
    
     # Print the number of documents in the collection to check if changes were made
    print(f"Number of documents in the database: {collection.count()}")

    #Confirm the embeddings were succesful
    msg = cl.Message(
        content=f"`{text_file.name}` embedded in the knowledge database! Ready to use. \n Documents on the database: `{get_document_prefixes(collection, delimiter)}`")
    msg.actions = actions
    await msg.send()
    
    
@cl.action_callback("Delete Document from DB") 
async def on_action(action):

    pref_list = get_document_prefixes(collection, delimiter)
    await cl.Message(
        content=f"List of documents embedded in the database: \n`{pref_list}").send()
    id = await cl.AskUserMessage(content="Choose document id you wish to delete from the list above", timeout=10).send()  
    id_to_delete = str(id['output']) 
    docs_to_delete = [doc_id for doc_id in collection.get().get('ids', []) if doc_id.startswith(id_to_delete)]
    if docs_to_delete:
            # Delete documents with the selected prefix
            collection.delete(ids=docs_to_delete)
            msg = cl.Message(content=f"Deleted documents with prefix: {id_to_delete} \n Number of documents in the database: {collection.count()} \n You can continue asking questions about the remaining documents")
            msg.actions = actions
            await msg.send()
            print(f"Deleted documents with prefix: {id_to_delete}")
    else:
            print(f"No documents found with prefix: {id_to_delete}")
            msg = cl.Message(content="No documents found with the selected prefix. Please choose a different prefix.")
            msg.actions = actions
            await msg.send()
        
    print(f"Number of documents in the database: {collection.count()} \n You can continue asking questions about the remaining documents")

@cl.action_callback("Clear Database") # deleting
async def on_action(action):
    global collection  # Use global keyword to access the global variable
    client = chromadb.PersistentClient(path='vector_db')
    client.delete_collection('pdf_docs')
    await cl.Message(content=f"Database was reset and is now empty. Re-opem the chatbot to initialize a new collection.").send()
    vector_db, collection = initialize_vector_database("vector_db", "pdf_docs", "vector_db", embeddings_framework, embeddings_model)