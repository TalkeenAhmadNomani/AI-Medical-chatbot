import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Optional: Load from .env file
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY is not set. Please export it or set it in your .env file.")

# ==== Step 1: Load raw PDF(s) ====
DATA_PATH = "data/"

def load_pdf_files(data_path):
    print("🔎 Loading PDF documents...")
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents/pages.")
    return documents

documents = load_pdf_files(DATA_PATH)


# ==== Step 2: Create Chunks ====
def create_chunks(extracted_docs):
    print("🔎 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_docs)
    print(f"✅ Created {len(text_chunks)} chunks.")
    return text_chunks

text_chunks = create_chunks(documents)


# ==== Step 3: Create Vector Embeddings ====
def get_embedding_model():
    print("🔎 Initializing OpenAI Embeddings model...")
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

embedding_model = get_embedding_model()


# ==== Step 4: Store embeddings in FAISS ====
DB_FAISS_PATH = "vectorstore/db_faiss"

print("💾 Creating FAISS index...")
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"✅ FAISS index saved at: {DB_FAISS_PATH}")
