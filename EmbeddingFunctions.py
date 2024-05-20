from langchain_community.document_loaders import PyPDFLoader
import chromadb
import ollama  # Import the missing module
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama


def initialize_vector_database(path, collection_name, persist_directory,framework, model):
    # Create a persistent client and a collection for the vector database
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} )

    # Load the previously stored documents into a Chroma vectorstore
    vectordb = Chroma(persist_directory=persist_directory,
                      client=client,
                      embedding_function=get_embeddings_model(framework, model),
                      collection_name=collection_name,
                      collection_metadata={"hnsw:space": "cosine"})

    # Get the initial count of documents in the collection and set a retriever
    initial_count = collection.count()
    print(f"Number of documents in the database: {collection.count()}")
    return vectordb, collection
    
def add_pdf_to_vectorstore_simple(text_file, id_str, collection, DELIMITER, embeddings_framework, embeddings_model):
    # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
    import ollama  # Import the missing module
    loader = PyPDFLoader(text_file.path)
    texts = [str(text) for text in loader.load_and_split()]
    for i, d in enumerate(texts):
        if embeddings_framework == "ollama":
            response = ollama.embeddings(model=embeddings_model, prompt=d)
            embedding = response["embedding"]
        elif embeddings_framework == "huggingface":
            model = HuggingFaceEmbeddings(model_name=embeddings_model)
            embedding = model.embed_query(d)
        else:
            print("this embedding framework is not implemented yet")
        id = id_str + DELIMITER + str(i)
        print(id)
        collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[d]
        )
            
def add_pdf_to_vectorstore_complete(text_file, id_str, collection, DELIMITER, embeddings_framework, embeddings_model, chunk_size, chunk_overlap):
    # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
        import ollama  # Import the missing module
        loader = PyPDFLoader(text_file.path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        texts = [str(text) for text in texts]
        for i, d in enumerate(texts):
            if embeddings_framework == "ollama":
                response = ollama.embeddings(model=embeddings_model, prompt=d)
                embedding = response["embedding"]
            elif embeddings_framework == "huggingface":
                model = HuggingFaceEmbeddings(model_name=embeddings_model)
                embedding = model.embed_query(d)
            else:
                print("this embedding framework is not implemented yet")
            id = id_str + DELIMITER + str(i)
            print(id)
            collection.add(
                ids=[id],
                embeddings=[embedding],
                documents=[d]
            )
            
def get_document_prefixes(collection, DELIMITER):
        # Get the documents in the collection
        documents = collection.get()
        ids = documents['ids']

        # Store id prefixes in a list
        pref_list = []
        for id in ids:
            prefix = id.split(DELIMITER)[0]
            if prefix not in pref_list:
                pref_list.append(prefix)
        return pref_list
    
def get_number_relevant_documents(vector_db, query, threshold):
    search_results = vector_db.similarity_search_with_score(query, 100000)
    #print(search_results)
    
    relevant_count = 0
    for document, score in search_results:
        print(f"Score: {score}")
        
        if score < threshold:
            relevant_count += 1
    
    # Return the count of relevant documents
    return relevant_count

def read_config(config_path: str) -> dict[str, str]:
    with open(config_path) as o:
        return yaml.safe_load(o)
    

def get_embeddings_model(framework: str, model: str):
    if framework == 'huggingface':
        return HuggingFaceEmbeddings(model_name=model)
    elif framework == 'ollama':
        return OllamaEmbeddings(model=model)
    else:
        raise NotImplementedError("...")


def get_chat_model(framework: str, model: str, temperature: int = None):
    if framework == 'huggingface':
        return HuggingFaceEmbeddings(model_name=model)
    elif framework == "ollama":
        return Ollama(model=model, temperature=0)
    else:
        raise NotImplementedError("...")
