
import chromadb
from chromadb.config import Settings
import chromadb
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import ollama

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdf_path", help="PDF document path", required=True)
parser.add_argument("-d", "--persist_directory", help="Path for stored database", required=True)
parser.add_argument("-c", "--collection", help="Collection name", required=True)
parser.add_argument("-id", "--doc_id", help="Document ID", required=True)
args = parser.parse_args() 

# Initialize the SentenceTransformer model
# Create an embedding function using the model
ef = OllamaEmbeddings(model="mxbai-embed-large")

vector_db = args.persist_directory
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_or_create_collection(name=args.collection)
initial_count = collection.count()
print(f"Initial count (before adding): {collection.count()}")

loader = PyPDFLoader(args.pdf_path)
texts = [str(text) for text in loader.load_and_split()]
for i, d in enumerate(texts):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
    embedding = response["embedding"]
    id=[args.doc_id + str(i)]
    print(id)
    collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[d]
    )


print(f"Loaded {len(texts)} documents from {args.pdf_path}")
print(f"Final count (after adding): {collection.count()}")

'''
results = collection.query(
    query_texts=["What are the responsabilities of the visitor at champalimaud according to the agreement?"],
    n_results=1
)

print(results)

collection.delete(ids=[id for id in ids if id.startswith("champs")])
print(collection.count()) # returns the number of items in the collection
'''