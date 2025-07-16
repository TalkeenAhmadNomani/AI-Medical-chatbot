# connect_memory_with_llm.py

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Optional: Load .env file if needed
# from dotenv import load_dotenv
# load_dotenv()

# Set your API key (can also use environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ==== Step 1: Load LLM ====
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY
    )

# ==== Step 2: Prompt Template ====
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know. Do not make up anything.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# ==== Step 3: Load FAISS DB ====
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ==== Step 4: Create QA Chain ====
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)

# ==== Step 5: Query and Error Handling ====
try:
    user_query = input("Write Query Here: ").strip()
    if not user_query:
        raise ValueError("Query cannot be empty.")

    response = qa_chain.invoke({'query': user_query})

    print("\n✅ RESULT:\n", response["result"])
    print("\n📄 SOURCE DOCUMENTS:")
    for doc in response["source_documents"]:
        print("-", doc.page_content)

except AssertionError:
    print("\n❌ Vector dimension mismatch!")
    print("Make sure your FAISS index was created using the same embedding model.")
    print("→ Recommended: Rebuild index using OpenAIEmbeddings.")

except ValueError as ve:
    print(f"\n⚠️ Input Error: {ve}")

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

except Exception as ex:
    print(f"\n🔥 Unexpected Error: {type(ex).__name__}: {ex}")
