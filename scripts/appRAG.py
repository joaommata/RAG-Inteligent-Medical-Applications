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
from EmbeddingFunctions import add_pdf_to_vectorstore, get_document_prefixes, initialize_vector_database

#Initialize Database -> path, collection_name, persist_directory
vector_db, collection = initialize_vector_database("vector_db", "pdf_docs", r"C:\Users\CCIG\joao_mata\vector_db")
retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.4})

# Define a template for the chat
llm = Ollama(model="llama3", temperature = 0)
template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say "I don't know". 
Don't make up an answer.
context: {context}
question: {question} according to the provided context?

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Starting an instance of the class
writer = ClassificationWriter()

DELIMITER = "://:"

# ------------------------------------STARTING THE CHAT----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

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
    msg = cl.Message(content="")
    documents = collection.get()
    print(documents['ids'])
# ------------------------------------HANDLING INCOMING MESSAGES-------------------------------------------
# ---------------------------------------------------------------------------------------------------------
actions = [
        cl.Action(name="Select pdf", value="upload_pdf", description="Upload PDF Documents"),
        cl.Action(name="Delete Document from DB", value="delete_doc", description="Choose a document to delete"),
        cl.Action(name="Good Answer", value="True", description="Provide Feedback"),
        cl.Action(name="Bad Answer", value="False", description="Provide Feedback")
    ]

# Define a function to be called when a message is received
@cl.on_message
async def on_message(message: cl.Message):
    # Create a retriever from the vectorstore. This seems to be a duplicate of the retriever created earlier and can be removed.    
    # Get the context from the user
    msg = cl.Message(content="")

    # Generate a random ID
    id_length = 10
    id_chars = string.ascii_uppercase + string.digits
    random_id = generate_random_id(id_length, id_chars)
    
    # Update chat history with the message content
    writer.update_chat_history(random_id, message.content)

    ''' Define a runnable pipeline for the chat. The pipeline takes a context from the retriever and a question from the message,
    # generates a prompt using the context and question, gets a response from the language model, and parses the response into a string. '''
    runnable = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Feedback buttons
    
    ai_response_content = []
    
     # Run the runnable pipeline asynchronously. This will generate a response from the language model for each question in the message content.
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        ai_response_content.append(chunk)
        await msg.stream_token(chunk)
        
    # Join the chunks of the AI response into a complete response and add it to the history
    complete_ai_response = ''.join(ai_response_content)
    writer.update_chat_history(random_id, complete_ai_response)
    
    # Get the documents relevant to the message, extract text, and update the context
    used_context = retriever.get_relevant_documents(msg.content)
    print(f"Number of retrieved documents: {len(used_context)}")
    #print(used_context) # to check
    text_from_documents = [doc.page_content for doc in used_context]
    writer.update_context(text_from_documents)
    
    # Set the actions for the message and write history, context and classifications to the JSON file
    msg.actions = actions
    writer.write()
    await msg.send()


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
    
    add_pdf_to_vectorstore(text_file, id_str, collection, DELIMITER)
    
     # Print the number of documents in the collection to check if changes were made
    print(f"Number of documents in the database: {collection.count()}")

    #Confirm the embeddings were succesful
    await cl.Message(
        content=f"`{text_file.name}` embedded in the knowledge database! Ready to use.").send()
    
    await cl.Message(
        content=f"Documents on the database: `{get_document_prefixes(collection, DELIMITER)}`").send()
    
    
@cl.action_callback("Delete Document from DB") 
async def on_action(action):

    pref_list = get_document_prefixes(collection, DELIMITER)
    await cl.Message(
        content=f"List of documents embedded in the database: \n`{pref_list}").send()
    id = await cl.AskUserMessage(content="Choose document id you wish to delete from the list above", timeout=10).send()  
    id_to_delete = str(id['output']) 
    docs_to_delete = [doc_id for doc_id in collection.get().get('ids', []) if doc_id.startswith(id_to_delete)]
    if docs_to_delete:
            # Delete documents with the selected prefix
            collection.delete(ids=docs_to_delete)
            await cl.Message(content=f"Deleted documents with prefix: {id_to_delete} \n Number of documents in the database: {collection.count()} \n You can continue asking questions about the remaining documents").send()
            print(f"Deleted documents with prefix: {id_to_delete}")
    else:
            print(f"No documents found with prefix: {id_to_delete}")
        
    print(f"Number of documents in the database: {collection.count()} \n You can continue asking questions about the remaining documents")
