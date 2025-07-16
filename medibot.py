import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ✅ Import our new utils
from logging_utils import setup_logger
from exception_utils import handle_exception

# === Setup ===
load_dotenv()
logger = setup_logger()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set!")
    st.stop()

DB_FAISS_PATH = "vectorstore/db_faiss"

# === Load FAISS Vectorstore ===
@st.cache_resource
def get_vectorstore():
    try:
        logger.info("Loading vectorstore...")
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vectorstore loaded successfully.")
        return db
    except Exception as e:
        handle_exception(e, logger)
        return None

# === Custom Prompt ===
def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say you don't know. Do not make up anything. 
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# === Streamlit App ===
def main():
    st.title("Ask Chatbot! (OpenAI GPT-4o-mini)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your question here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    openai_api_key=OPENAI_API_KEY,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            source_texts = "\n".join([f"- {doc.page_content}" for doc in source_documents])
            result_to_show = f"{result}\n\n**Source Documents:**\n{source_texts}"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            logger.info("Answer delivered to user.")

        except Exception as e:
            handle_exception(e, logger)

if __name__ == "__main__":
    main()
