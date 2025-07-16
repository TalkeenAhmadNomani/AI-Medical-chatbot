# connect_memory_with_llm.py

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from logging_utils import setup_logger
from exception_utils import handle_exception

# === Setup ===
load_dotenv()
logger = setup_logger()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY is not set!")
    raise EnvironmentError("OPENAI_API_KEY must be set in your environment or .env file.")

DB_FAISS_PATH = "vectorstore/db_faiss"

# === Step 1: Load LLM ===
def load_llm():
    logger.info("🔁 Loading GPT-4o-mini LLM...")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY
    )

# === Step 2: Prompt Template ===
def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say you don't know. Do not make up anything.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# === Main Execution ===
def main():
    try:
        logger.info("📦 Loading FAISS index and embedding model...")
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

        logger.info("🔗 Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        user_query = input("📝 Write Query Here: ").strip()
        if not user_query:
            raise ValueError("Query cannot be empty.")

        logger.info("🧠 Querying LLM...")
        response = qa_chain.invoke({'query': user_query})

        print("\n✅ RESULT:\n", response["result"])
        print("\n📄 SOURCE DOCUMENTS:")
        for doc in response["source_documents"]:
            print("-", doc.page_content)

    except AssertionError:
        print("\n❌ Vector dimension mismatch!")
        print("Make sure your FAISS index was created using the same embedding model.")
        logger.error("Vector dimension mismatch with FAISS index.")

    except ValueError as ve:
        print(f"\n⚠️ Input Error: {ve}")
        logger.warning(f"Input Error: {ve}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        logger.warning("Execution interrupted by user.")

    except Exception as e:
        handle_exception(e, logger)


if __name__ == "__main__":
    main()
