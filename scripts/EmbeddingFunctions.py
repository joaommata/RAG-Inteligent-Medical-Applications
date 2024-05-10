from langchain_community.document_loaders import PyPDFLoader
import chromadb
import ollama  # Import the missing module
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter



def initialize_vector_database(path, collection_name, persist_directory):
    # Create a persistent client and a collection for the vector database
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} )

    # Load the previously stored documents into a Chroma vectorstore
    vectordb = Chroma(persist_directory=persist_directory,
                      client=client,
                      embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
                      collection_name=collection_name,
                      collection_metadata={"hnsw:space": "cosine"})

    # Get the initial count of documents in the collection and set a retriever
    initial_count = collection.count()
    print(f"Number of documents in the database: {collection.count()}")
    return vectordb, collection
    
def add_pdf_to_vectorstore(text_file, id_str, collection, DELIMITER):
        # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
        import ollama  # Import the missing module
        loader = PyPDFLoader(text_file.path)
        '''
        text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    is_separator_regex=False,)

        texts = text_splitter.split_documents(documents)
         '''
        texts = [str(text) for text in loader.load_and_split()]
        for i, d in enumerate(texts):
            response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
            embedding = response["embedding"]
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
    