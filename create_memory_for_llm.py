import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from logging_utils import setup_logger
from exception_utils import handle_exception

# === Setup ===
load_dotenv()

logger = setup_logger()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY is not set in environment or .env file!")
    raise EnvironmentError("OPENAI_API_KEY is not set.")


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


# === Load raw PDFs ===
def load_pdf_files(data_path):
    logger.info("🔎 Loading PDF documents...")
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    logger.info(f"✅ Loaded {len(documents)} documents/pages.")
    return documents


# === Split into chunks ===
def create_chunks(extracted_docs):
    logger.info("🔎 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_docs)
    logger.info(f"✅ Created {len(text_chunks)} chunks.")
    return text_chunks


# === Get embedding model ===
def get_embedding_model():
    logger.info("🔎 Initializing OpenAI Embeddings model...")
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# === Main index build ===
def main():
    try:
        documents = load_pdf_files(DATA_PATH)
        text_chunks = create_chunks(documents)
        embedding_model = get_embedding_model()

        logger.info("💾 Creating FAISS index...")
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        logger.info(f"✅ FAISS index saved at: {DB_FAISS_PATH}")

    except Exception as e:
        handle_exception(e, logger)


if __name__ == "__main__":
    main()
